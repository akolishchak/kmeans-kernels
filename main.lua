--
--  main.lua
--  kmeans-kernels
--
--  Created by Andrey Kolishchak on 12/19/15.
--
require 'nn'
require 'optim'
require 'image'
require 'image'
require 'patches'
require 'zca'
require 'kmeans'
require 'data.dataset'

cmd = torch.CmdLine()
cmd:text()
cmd:text('k-means cnn kernels')
cmd:text()
cmd:text('Options')
cmd:option('-pass_type', 2, '0 - linear training, 1 - cnn training, 2 - kmeans cnn')
cmd:option('-kernel_width', 15, 'kernel width')
cmd:option('-kernel_num', 8, 'number of kernels')
cmd:option('-whitening', 1, 'data whitening')
cmd:option('-gpu',2,'0 - cpu, 1 - cunn, 2 - cudnn')
cmd:option('-patch_num', 50000, 'snumber of patches')
cmd:option('-epsilon', 1e-5, 'zca epsilon')
cmd:option('-learning_rate',1e-4,'learning rate')
cmd:option('-batch_size',100,'batch size')
cmd:option('-max_epoch',100,'number of passes through the training data')
cmd:option('-kmeans_iterations',100,'number of k-means iterations')
cmd:option('-output_path','images','path for output images')


local opt = cmd:parse(arg)

if opt.gpu > 0 then
  require 'cunn'
  if opt.gpu == 2 then
    require 'cudnn'
  end
end

--
-- load data
--
local dataset = load_mnist(opt)

-- white data
if opt.whitening == 1 then
  print("whitening...")
  dataset.train_x, zca_params = zca(dataset.train_x:double(), opt.epsilon)
  dataset.test_x, _ = zca(dataset.test_x:double(), opt.epsilon, zca_params)
  if opt.gpu > 0 then
    dataset.train_x = dataset.train_x:cuda()
    dataset.test_x = dataset.test_x:cuda()
  end
  collectgarbage(); collectgarbage()
end

local centroids

if opt.pass_type >= 2 then
  
  print("clustering...")
  --
  -- get patches from original images
  --
  local patches = get_patches(dataset.train_x, opt.patch_num, opt.kernel_width)

  --
  -- form clusters
  --
  centroids = dataset.train_x.new(opt.kernel_num, dataset.train_x:size(2),        opt.kernel_width*opt.kernel_width)

  for channel=1,dataset.train_x:size(2) do
    centroids[{{}, channel, {}}] = kmeans(patches[{{}, channel, {}}], opt.kernel_num, opt.kmeans_iterations)
  end

end
  
--
-- convolutional model
--
local pad = ( opt.kernel_width - 1 ) / 2
local conv = nn.SpatialConvolution(dataset.train_x:size(2), opt.kernel_num, opt.kernel_width, opt.kernel_width, 1, 1, pad, pad)
  

local conv_model = nn.Sequential()
conv_model:add(conv)
conv_model:add(nn.ELU())
conv_model:add(nn.Reshape(opt.kernel_num*dataset.train_x:size(3)*dataset.train_x:size(4)))

--
-- linear classifier
--
local class_model = nn.Sequential()
class_model:add(nn.Linear(opt.kernel_num*dataset.train_x:size(3)*dataset.train_x:size(4), 10))
class_model:add(nn.LogSoftMax())

local model = nn.Sequential()
model:add(conv_model)
model:add(class_model)

local criterion = nn.ClassNLLCriterion()


if opt.gpu > 0 then
  model:cuda()
  criterion:cuda()
  
  if opt.gpu == 2 then
    cudnn.convert(model, cudnn)
    cudnn.convert(criterion, cudnn)
    cudnn.benchmark = true
  end
end

if opt.pass_type >= 2 then
  --
  -- init weights of cnn
  --
  conv.weight:copy(centroids:view(opt.kernel_num,-1))
  conv.bias:zero()
end

local params, grad_params = model:getParameters()

local conv_weights = conv.weight:view(-1, opt.kernel_width)

--
-- optimize
--
local iterations = opt.max_epoch*dataset.train_x:size(1)/opt.batch_size
local batch_start = 1

function feval(x)
  if x ~= params then
        params:copy(x)
  end
  grad_params:zero()
  
  -- load batch
  local input = dataset.train_x[{{batch_start, batch_start+opt.batch_size-1},{}}]
  local target = dataset.train_y[{{batch_start, batch_start+opt.batch_size-1}}]
    
  -- forward pass
  local conv_input = conv_model:forward(input)
  local output = class_model:forward(conv_input)
  local loss = criterion:forward(output, target)
  
  -- back prop
  local dloss_doutput = criterion:backward(output, target)
  local dloss_dclass = class_model:backward(conv_input, dloss_doutput)
  
  if opt.pass_type == 1 then
    conv_model:backward(input, dloss_dclass)
  end
  
  return loss, grad_params
end

-- train
class_model:training()

local optim_state = {learningRate = opt.learning_rate}

print("trainig...")

for it = 1,iterations do
  
    local _, loss = optim.adam(feval, params, optim_state)
    
    if it % 100 == 0 then
      print(string.format("batch = %d, loss = %f", it, loss[1]))
    end
  
    batch_start = batch_start + opt.batch_size
    if batch_start > dataset.train_x:size(1) then
      batch_start = 1
    end 
    
end

class_model:evaluate()

print("evaluation...")
paths.mkdir(opt.output_path)

function get_loss(x, y, log_fails)
  local match = 0.0
  for i=1,x:size(1),opt.batch_size do  
    local input = x[{{i, i+opt.batch_size-1},{}}]
    local target = y[{{i, i+opt.batch_size-1}}]

    local conv_input = conv_model:forward(input)
    local output = class_model:forward(conv_input)
    prob, idx = torch.max(output, 2)
    
    match = match + torch.mean(idx:eq(target):float())/(x:size(1)/opt.batch_size)
    
    local matches = idx:eq(target)
    if log_fails == true then
      for j=1,matches:size(1) do
        if matches[j][1] == 0 then
          local k = i-1+j
          image.save(opt.output_path..'/fail_'..tostring(k)..'-'..tostring(y[k])..'-'..tostring(idx[j][1])..'.jpg',x[{{k},{}}]:view(28,28))
        end 
      end
    end
    
  end

  return match
end


print(string.format("training = %.2f%%, testing = %.2f%%", get_loss(dataset.train_x, dataset.train_y, false)*100.0, get_loss(dataset.test_x, dataset.test_y, false)*100.0))

image.save(opt.output_path..'/weights-'..tostring(opt.pass_type)..'.jpg', conv_weights)

