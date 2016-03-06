--
--  zca.lua
--  kmeans-kernels
--
--  Created by Andrey Kolishchak on 12/19/15.
--
function zca(x, epsilon, params)
  
  local x_flat = x:view(x:size(1), x:size(2), -1)
  
  if params == nil then
    params = {}
    for i=1,x_flat:size(2) do
      
      local data = x_flat[{{},i,{}}]
      local sigma = data:t() * data / x:size(1)
      
      local channel_params = {}
      channel_params.u, channel_params.s, _ = torch.svd(sigma)
      params[i] = channel_params
    end
  end
  
  local x_zca = x_flat.new(x_flat:size())
  
  for i=1,x_flat:size(2) do
    
    local x_rot = x_flat[{{},i,{}}] * params[i].u
  
    local x_pca = x_rot * torch.diag(torch.ones(params[i].s:size()):cdiv(torch.sqrt(params[i].s + epsilon)))
    x_zca[{{},i,{}}] = x_pca * params[i].u:t()
  end
    
  return x_zca:viewAs(x), params
end
