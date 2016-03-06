--
--  kmeans.lua
--  kmeans-kernels
--
--  Created by Andrey Kolishchak on 12/19/15.
--

function kmeans(x, c_num, max_iter)
  
  max_iter = max_iter or 1
  
  local centroids = x.new(c_num, x:size(2))--:normal()
    
  --
  -- initialize centroids 
  --
  centroids:normal(x:mean(), x:std())
  local dist = x.new(x:size(1), c_num)

  for iter = 1,max_iter do
    
    print(string.format("kmeans iteration = %d", iter))
    --
    -- find closest centroids
    --
    for c = 1,c_num do
      local centroid = torch.repeatTensor(centroids[c], x:size(1), 1)
      dist[{{},{c}}] = torch.norm(x - centroid, 2, 2)
    end
    collectgarbage(); collectgarbage()
  
    _, centroid_idx = torch.min(dist, 2)
    centroid_idx:resize(centroid_idx:size(1))
    --
    -- compute new centroids
    --
    for c = 1,c_num do
      idx = centroid_idx:eq(c):float():nonzero()
      if idx:dim() ~= 0 then
        centroids[c] = torch.mean(x:index(1, idx:view(idx:size(1))), 1)
      else
        --centroids[c] = 
      end
    end
  end
  
  return centroids
  
end
