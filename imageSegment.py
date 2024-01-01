import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageSegmentNet(nn.Module):
    def __init__(self, n_xsegments, n_ysegments):
        super(ImageSegmentNet, self).__init__()
        self.n_xsegments = n_xsegments
        self.n_ysegments = n_ysegments

        #Initialize x coordinates between 0-1 so segments are equal size
        self._custom_init(0, 1, 0, 1)

    def _custom_init(self, x_min, x_max, y_min, y_max):
        #Initialize x coordinates between so segments are equal size
        x_init = torch.Tensor(np.linspace(x_min, x_max, self.n_xsegments + 1))
        # make a 2D init tensor
        x_2dinit = torch.Tensor(self.n_ysegments + 1, self.n_xsegments + 1)
        # Update the x_2dinit tensor with the repeated values
        x_2dinit[:] = x_init.unsqueeze(0)
        # initialize all rows of x
        self.x = nn.Parameter(torch.Tensor(x_2dinit))

        y_init = torch.Tensor(np.linspace(y_min, y_max, self.n_ysegments + 1))
        # make a 2D init tensor
        y_2dinit = torch.Tensor(self.n_ysegments + 1, self.n_xsegments + 1)
        # Update the x_2dinit tensor with the repeated values
        y_2dinit[:] = y_init.unsqueeze(1)
        # initialize all rows of y
        self.y = nn.Parameter(torch.Tensor(y_2dinit))
    
    def set_image(self, bw_array):
        [self.height, self.width] = bw_array.shape
        self.bw_array = bw_array
        self.set_image_size(self.height, self.width)

    def set_image_size(self, height, width):
        xmax = width + 1
        ymax = height + 1
        maxval = max(xmax, ymax)
        self._custom_init(0., xmax/maxval, 0., ymax/maxval)
        self.x_in = torch.arange(1./maxval, xmax/maxval, 1/maxval)
        self.y_in = torch.arange(1./maxval, ymax/maxval, 1/maxval)
        self.maxval = maxval #store as we will need it to denormalize
        #calculate actual y so we can calculate loss
        self._calc_yact()
        self._calc_xact()

    def _calc_yact(self):
        """ Calculates the y values after shringing y based on n_ysegments
            this values is what ypred is predicting
        """
        #create array that contains y pos values instead of 1
        yval_array = self.y_in.view(self.y_in.shape[0], 1)*self.bw_array
        #reshape the array such that height is n_ysegments
        yval_array_3d = yval_array.view(self.n_ysegments,
                                int(self.height/self.n_ysegments), self.width)
        
        # Calculate the minimum value that is > 0 from dim=1
        # we have to do this in 3 steps due to 0 values
        
        #step1: Replace 0 with float('inf') in-place:
        yval_array_3d[yval_array_3d == 0] = float('inf')
        
        # step2: Find the minimum along dim=1
        self.yact, _ = torch.min(yval_array_3d, dim=1)

        #Step3: replace inf with zeros again
        self.yact[self.yact == float('inf')] = 0

    def _calc_xact(self):
        """ Calculates the x values after shringing x based on n_xsegments
            this values is what xpred is predicting
        """
        #create array that contains x pos values instead of 1
        xval_array = self.x_in*self.bw_array
        #reshape the array such that height is n_ysegments
        #TODO: figure out a way to avoid rounding in int()
        xval_array_3d = xval_array.view(self.height,self.n_xsegments,
                    int(self.width/self.n_xsegments))
        
        # Calculate the minimum value that is > 0 from dim=1
        # we have to do this in 3 steps due to 0 values
        
        #step1: Replace 0 with float('inf') in-place:
        xval_array_3d[xval_array_3d == 0] = float('inf')
        
        # step2: Find the minimum along dim=1
        self.xact, _ = torch.min(xval_array_3d, dim=2)

        #Step3: replace inf with zeros again
        self.xact[self.xact == float('inf')] = 0

    def _calc_ymask(self):
        """ Calculate the mask required due to segmentation of X and Y
            Due to segmentation a continuous value of X predicts a discrete Y
            and a continuous Y predicts a discrete X within a segment.
        """
        # expand y to a new dimension
        y_in_4d = self.y_in.unsqueeze(1).unsqueeze(1)
        #fit/average y_in to y_segment
        lt = torch.lt(y_in_4d, self.y[1:,:-1])
        ge = torch.ge(y_in_4d, self.y[:-1,:-1])
        self.ymask = (lt & ge)
        #take care of edge cases
        mask_lt = torch.lt(y_in_4d, self.y[0:1,:-1])
        mask_ge = torch.ge(y_in_4d, self.y[-1:,:-1])
        # then do OR with mask so these are included for prediction.
        self.ymask[:,-1:,:] = self.ymask[:,-1:,:] | mask_ge
        self.ymask[:,:1, :] = self.ymask[:,:1, :] | mask_lt
        
        # TODO : can we shrink the ymask as below - test it
        # model.ymask.sum(dim=2).gt(0) - NO WE CANNOT
        # Explaination:
        # ymask shape of say torch.Size([20, 10, 14]) for a 20x28 image
        # and xsegment=14, ysegment=10 means
        # for a height of 20 it can be split in 10 groups.
        # each column (out of 28) can be group together a 14 similar ones
        # but each of the 14 may have different 10 y values.
        # so we cannot eliminate any dimension.

    def _calc_xmask(self):
        """ Calculate the mask required due to segmentation of X and Y
            Due to segmentation a continuous value of X predicts a discrete Y
            and a continuous Y predicts a discrete X within a segment.
        """
        # expand x to a new dimension
        x_in_4d = self.x_in.unsqueeze(1).unsqueeze(1)
        #fit/average y_in to y_segment
        lt = torch.lt(x_in_4d, self.x[:-1,1:])
        ge = torch.ge(x_in_4d, self.x[:-1,:-1])
        self.xmask = (lt & ge)
        #take care of edge cases
        mask_lt = torch.lt(x_in_4d, self.x[:-1,0:1])
        mask_ge = torch.ge(x_in_4d, self.x[:-1,-1:])
        # then do OR with mask so these are included for prediction.
        self.xmask[:,:,-1:] = self.xmask[:,:,-1:] | mask_ge
        self.xmask[:,:, :1] = self.xmask[:,:, :1] | mask_lt

    def _predict_y(self):
        # When we predict y each segment makes its prediction
        # mask allows only the correct segment to predict masking out the rest.
        self._calc_xmask()
        # Ratio is the segment (y2-y1)/(x2-x1) ratio 
        # consider each segment as a right triangle so we can find new y-y1 as ratio of x-x1
        #The ratio can get -inf or inf. we need to protect against it
        divider = (self.x[:-1,1:]-self.x[:-1,:-1])
        divider[divider == 0.] = 0.0001
        self.yratio = (self.y[:-1,1:]-self.y[:-1,:-1])/divider
       
        # yratio.shape = [n_ysegments, n_xsegments]
        x_in_4d = self.x_in.unsqueeze(1).unsqueeze(1)
        self.ypred = (x_in_4d - self.x[:-1,:-1])*self.xmask*self.yratio + self.xmask * self.y[:-1,:-1]
        # remove extra dimension as we should only have one value and rest 0 due to mask
        self.ypred = self.ypred.sum(2)
        # we need to reshape Y TODO: Why? can we fix this from the top?
        self.ypred = self.ypred.permute(1,0)

    
    def _predict_x(self):
        # When we predict x each segment makes its prediction
        # mask allows only the correct segment to predict masking out the rest.
        self._calc_ymask()
        # Ratio is the segment (y2-y1)/(x2-x1) ratio 
        #The ratio can get -inf or inf. we need to protect against it
        divider = (self.y[1:,:-1]-self.y[:-1,:-1])
        divider[divider == 0.] = 0.0001
        # consider each segment as a right triangle so we can find new y-y1 as ratio of x-x1
        self.xratio = (self.x[1:,:-1]-self.x[:-1,:-1])/divider

        y_in_4d = self.y_in.unsqueeze(1).unsqueeze(1)
        self.xpred = (y_in_4d - self.y[:-1,:-1])*self.ymask*self.xratio + self.ymask * self.x[:-1,:-1]
        # remove extra dimension as we should only have one value and rest 0 due to mask
        self.xpred = self.xpred.sum(1)
    
    def create_yimage(self):
        # create yimage by following steps of calc_yact but populate values
        with torch.no_grad():
            self.yimage = torch.zeros_like(self.bw_array).to(torch.float32)
            yimage_3d = self.yimage.view(self.n_ysegments,
                                    int(self.height/self.n_ysegments), self.width)
            yimage_3d[:,:,:] = self.ypred.unsqueeze(dim=1)

    def create_ximage(self):
        # create ximage by following steps of calc_xact but populate values
        with torch.no_grad():
            self.ximage = torch.zeros_like(self.bw_array).to(torch.float32)
            #for some reason view didn't work so reshape
            ximage_3d = self.ximage.reshape(self.height,self.n_xsegments,
                    int(self.width/self.n_xsegments))
            ximage_3d[:,:,:] = self.xpred.unsqueeze(dim=2)
            self.ximage = ximage_3d.view(self.height, self.width)

    def forward(self):
        # When we predict y each segment makes its prediction
        self._predict_y()
        #self._predict_x()
        #return self.xpred
        #self._predict_x()
        #self._image_from_y()
        #self._image_from_x()
        #self.img_out = torch.logical_or(self.ximage, self.yimage).to(torch.float32)
        #return self.ximage