# Segment-v3
import numpy as np
import torch
import torch.nn as nn

class Segment(nn.Module):
    r"""Divides incoming data into segment_features then applies two point
    formula to each segment.
    For any given x only one segment calculates y all others are masked out.
    All in_features (x) contributes in calculation of ever out_feature (y)
    but each combination of in_feature and out out_feature has its own
    set of segments. This will create total number of
    parameters = out_features*in_features*(segment_features + 1)
    
        :math:`y = (y2 - y1)/(x2 - x1)*(x - x1) + y1`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        segment_features: number of segments on x dimensions

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
        dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
        are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        x: the learnable x of the module of shape
            :math:`(\text{in\_features})`, \text{segment\_features + 1}, 
                    \text{out\_features})`.
            The values are
            initialized from :math:`(x.min(), x.max())` and intermediate
            x are linearly spaced.

    Examples::

        >>> m = Segment(20, 30, 10)
        >>> input = torch.randn(128, 20)
        >>> m.custom_init(x.min, x.max)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features', 'segment_features']
    in_features: int
    out_features: int
    segment_features: int
    x: torch.Tensor
    y: torch.Tensor
    def __init__(self, in_features, out_features, segment_features):
        super(Segment, self).__init__()
        self.segment_features = segment_features
        self.out_features = out_features
        self.in_features = in_features

        self.x = nn.Parameter(torch.Tensor(in_features,
                                            segment_features + 1, out_features))
        self.y = nn.Parameter(torch.Tensor(in_features,
                                            segment_features + 1, out_features))
        # As this is BW image let us hardcode for now
        self.custom_init(0,1)
        self.initialized = True

    def custom_init(self, x_min, x_max):
        x_init = torch.empty_like(self.x)
        for i in np.arange(self.in_features):
            for j in np.arange(self.out_features):
                x_init[i, :, j] = torch.linspace(x_min, x_max,
                                    self.segment_features + 1)
        # initialize all rows of x
        self.x = nn.Parameter(torch.Tensor(x_init))
    
    def _calc_mask(self, x_in):
        # x_in.shape = [N, in_features]
        self.x_in_4d =  x_in.unsqueeze(-1).unsqueeze(-1)     
        # x_in_4d.shape =  [N, in_features, 1, 1] 
        # to find out if x is between segments
        lt = torch.lt(self.x_in_4d, self.x[:, 1:, :]) 
        ge = torch.ge(self.x_in_4d, self.x[:, :-1, :]) 
        self.mask = (lt & ge)
        #mask.shape = [N, in_features, segment_features, out_features]

        # This mask doesn't include x where x is below 1st segment start
        # or after last segment end
        # we create new mask to capture the x values beyond segments
        mask_lt = torch.lt(self.x_in_4d,self.x[:, 0:1, :]) #x less than 1st segment
        # then do OR with mask so these are included for prediction.
        self.mask[:,:, 0:1, :] = self.mask[:, :, 0:1, :] | mask_lt

        #do the same for last x of segment
        mask_ge = torch.ge(self.x_in_4d,self.x[:, -1:, :]) 
        self.mask[:,:, -1:, :] = self.mask[:, :, -1:, :] | mask_ge

    def forward(self, x_in):
        self._calc_mask(x_in)

        divider = (self.x[:,1:,:]-self.x[:,:-1,:])
        #The ratio can get -inf or inf. we need to protect against it
        divider[divider == 0.] = 0.0001

        # Ratio is the segment (y2-y1)/(x2-x1) ratio 
        ratio = (self.y[:,1:,:]-self.y[:,:-1,:])/divider

        ypred = ratio*self.mask*(self.x_in_4d - self.x[:,:-1,:]) + self.mask * self.y[:,:-1,:]
        # ypred.shape = [N, in_features, segment_features, out_features]
        # we can sum up by in_features (as y is cumulative of all f(x))
        # and sum by segment_features (only one segment should be non zero)
        ypred = torch.sum(ypred, dim=(1,2))
        return ypred
            
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, segment_features={self.segment_features}'