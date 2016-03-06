--
--  patches.lua
--  kmeans-kernels
--
--  Created by Andrey Kolishchak on 12/19/15.
--

function get_patches(data, num, width)
  max_width = data:size(3) - width + 1
  max_heigth = data:size(4) - width + 1
  
  local patches = data.new(num, data:size(2), width*width)
  local img_num = torch.randperm(data:size(1))
  
  for channel=1,data:size(2) do
    for i=1,num do
      local x = math.random(max_heigth)
      local y = math.random(max_width)
      local img = data[img_num[i]][channel]
      local patch = img[{{x,x+width-1},{y,y+width-1}}]
      patches[i][channel] = patch
    end
  end
  
  
  return patches
end
