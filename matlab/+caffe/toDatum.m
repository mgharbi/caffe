function  datum = toDatum( varargin )
%FROMDATUM decode image and label from caffe protobuf.

CHECK(nargin > 0, ['usage: '...
    'datum = fromDatum( img, label)']); 
img = varargin{1};
if numel(varargin) < 2
    label = 0;
end

datum = caffe_('to_datum', img, label);

end
