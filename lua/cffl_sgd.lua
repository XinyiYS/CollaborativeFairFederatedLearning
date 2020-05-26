require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides a normalization operator
require 'optim'   -- an optimization package, for online and batch methods
require 'dataset-mnist'
require 'dataset-svhn'
os.execute('mkdir ' .. 'save')

function isnan(x) return x ~= x end
function bell_curve(x) return math.sin(math.pi*x) end
-- sigmoid curve: accuracy_combined / (accuracy_combined+accuracy_leaveone)=alpha/(2*alpha+gap), gap>0, - contribution; gap<0, + contribution
-- function credit_curve(x) return 1/(1+math.exp(-15*(x-0.5))) end
function credit_curve(x) return math.sinh(5*x) end

function lapnoise(l, n) 
   local noise = - torch.rand(n):log():mul(l)
   local sign = torch.rand(n):apply(function(x) if x < 0.5 then return 1 else return -1 end end)
   return noise:cmul(sign)
end

function cut(value, range)
   local cvalue = value
   if range > 0 then
      if cvalue < -range then
         cvalue = -range
      elseif cvalue > range then
         cvalue = range
      end
   end
   return cvalue
end

-- options
cmd = torch.CmdLine()

cmd:option('-dataset',           'mnist', 'svhn | mnist')
cmd:option('-dataSizeFrac',      1,       'the fraction of dataset to be used for training')
cmd:option('-model',             'cvn',   'convnet | cvn | linear | mlp | deep')
cmd:option('-method',            'seq',   'seq | fla | asy |syn')
-- cmd:option('-plevel',            1,      '1 | 5')
cmd:option('-IID',            1,      '1 | 0')
-- simulate party 1: 100 records, other 600 
cmd:option('-imbalanced',        0,      '0 | 1')
cmd:option('-learningRate',      1e-3,    '')
cmd:option('-learningRateDecay', 1e-7,    '')
cmd:option('-batchSize',         1,       '')
cmd:option('-weightDecay',       0,       '')
cmd:option('-momentum',          0,       '')
cmd:option('-threads',           2,       '')
cmd:option('-netSize',           30,      '')
cmd:option('-shardSizeFrac',     0.01,    'fraction of the training set in each shard')
cmd:option('-uploadFraction',    0.1,     'fraction of parameters to be uploaded after training')
cmd:option('-downloadFraction',  1,       'fraction of parameters to be downloaded before training')
cmd:option('-epochFraction',     1,       '')
cmd:option('-epsilon',           0,       'epsilon for dp. 0: disable it')
cmd:option('-delta',             0,       'delta for dp.')
cmd:option('-range',             0.001,       'cut the gradiants between -range and range. 0: disable it')
cmd:option('-threshold',         0,       'release those whose abs value is greater than threshold. 0: disable it')
cmd:option('-nepochs',           60,     '')
cmd:option('-local_nepochs',     1,     '')
cmd:option('-taskID',            '0',     'the ID associated to the task')
cmd:option('-folder',            'save',  '')
-- record the indices of trainshard, ensure same dataset comparison for different framework
-- mnist_p4
-- mnist_p15
-- mnist_p30
-- mnist_p50
-- mnist_p4_imbalanced
-- mnist_p15_imbalanced
-- mnist_p30_imbalanced
-- mnist_p50_imbalanced
cmd:option('-shardID',            '0',     'the ID associated to the shardfile')
cmd:option('-run',            '0',  '')
cmd:option('-credit_thres',      1,  '0 | 1')
cmd:option('-credit_fade',   1,  '0 | 1')
cmd:option('-update_criteria',   'large',  'large | random')
cmd:option('-pretrain',   0,  '0|2|5|10')
-- cmd:option('-pretrain_epochs',   10,  '10 | 5')
cmd:option('-alpha',   5,  '1| 5 | 8| 10')

opt = cmd:parse(arg or {})

shardfile=paths.concat(opt.folder, 'trainshard.' .. opt.shardID .. '.' .. opt.run) 
print(shardfile)
val_shardfile=paths.concat(opt.folder, 'valshard.' .. opt.shardID .. '.' .. opt.run) 
print(val_shardfile)
epochfile = paths.concat(opt.folder, 'epoch.' .. opt.taskID .. '.'  .. opt.run)
resultsfile = paths.concat(opt.folder, 'results.' .. opt.taskID .. '.'  .. opt.run)
-- pfile = paths.concat(opt.folder, 'p.' .. opt.taskID .. '.'  .. opt.run)
psfile = paths.concat(opt.folder, 'ps.' .. opt.taskID .. '.'  .. opt.run)
paramfile = paths.concat(opt.folder, 'parameters.' .. opt.taskID .. '.'  .. opt.run)
gradstatfile = paths.concat(opt.folder, 'gradstat.' .. opt.taskID .. '.'  .. opt.run)
gradstatpfile = paths.concat(opt.folder, 'gradstatp.' .. opt.taskID .. '.'  .. opt.run)
nupdatesfile = paths.concat(opt.folder, 'nupdates.' .. opt.taskID .. '.'  .. opt.run)
pointfile = paths.concat(opt.folder, 'point.' .. opt.taskID .. '.'  .. opt.run)
creditfile = paths.concat(opt.folder, 'credit.' .. opt.taskID .. '.'  .. opt.run)
-- plevelfile = paths.concat(opt.folder, 'privacy_level.' .. opt.taskID .. '.'  .. opt.run)
-- plevelfile = paths.concat(opt.folder, 'privacy_level.' .. opt.shardID .. '.'  .. opt.run)
-- print(plevelfile)
uploadfile = paths.concat(opt.folder, 'upload.' .. opt.taskID .. '.'  .. opt.run)
downloadfile = paths.concat(opt.folder, 'download.' .. opt.taskID .. '.'  .. opt.run)
iaccfile = paths.concat(opt.folder, 'iacc.' .. opt.taskID .. '.'  .. opt.run)
saccfile = paths.concat(opt.folder, 'sacc.' .. opt.taskID .. '.'  .. opt.run)
print(opt)
psfile_pretrain = paths.concat(opt.folder, 'ps'..opt.pretrain .. '.' .. opt.taskID .. '.'  .. opt.run)
psfile_pretrain2 = paths.concat(opt.folder, 'ps2.' .. opt.taskID .. '.'  .. opt.run)
psfile_pretrain5 = paths.concat(opt.folder, 'ps5.' .. opt.taskID .. '.'  .. opt.run)
psfile_pretrain10 = paths.concat(opt.folder, 'ps10.' .. opt.taskID .. '.'  .. opt.run)
standalone_acc_file = paths.concat(opt.folder, 'standalone_acc' .. opt.taskID .. '.'  .. opt.run)

-- config torch
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')


-- create training set and test set
if opt.dataset == 'mnist' then
  nbTrainingPatches = 60000
  nbTestingPatches = 10000
  geometry = {32, 32}
  trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
  testData = mnist.loadTestSet(nbTestingPatches, geometry)
  local mean = trainData.data:mean()
  local std = trainData.data:std()
  trainData.data:add(-mean):div(std)
  testData.data:add(-mean):div(std)
end
if opt.dataset == 'svhn' then
  trainData, testData = svhn.loadData()
  -- Preprocesss
  local mean = trainData.data:mean()
  local std = trainData.data:std()
  trainData.data:add(-mean):div(std)
  testData.data:add(-mean):div(std)
end
trainSize = math.ceil(opt.dataSizeFrac * trainData.labels:size(1))
testSize  = testData.labels:size(1)
print('testSize: '..testSize)

trainData.size = function() return trainSize end
testData.size = function() return testSize end

val_size=math.ceil(0.1*trainSize)
-- test 1000 from test set,  randomly shuffle the test set to ensure that the different classes are balanced
local test_shuffle = torch.randperm(testSize)

shardSize={}
trainData.shard = {}
if paths.filep(shardfile) and paths.filep(val_shardfile) then
   trainData.shard= torch.load(shardfile, 'binary')
   val_shard = torch.load(val_shardfile, 'binary')
   print('load shard indices ' .. shardfile)
   for nid = 1, opt.netSize do
      shardSize[nid] = #trainData.shard[nid]
      shardSize[nid] = shardSize[nid][1]
   end
else
  print('create shard indices')
  -- non-overlap between val and each party's train data
  local val_shffl = torch.randperm(trainData:size())
  -- val_shard = {}
  val_shard = val_shffl[{ {1,val_size} }]
  torch.save(val_shardfile, val_shard)
  train_shard=val_shffl[{ {val_size+1,trainData:size()} }]
  if opt.imbalanced==0 and opt.IID==1 then
    local shffl = torch.randperm(trainData:size()-val_size)
    -- local shffl = torch.randperm(#train_shard)
    accessed=1
    for nid = 1, opt.netSize do
      shardSize[nid] = math.ceil(opt.shardSizeFrac * trainSize)
      trainData.shard[nid] = shffl[{ {accessed,accessed+shardSize[nid]-1} }]
      accessed=accessed+shardSize[nid]
      print('balanced shardSize for party '.. nid .. ': ' .. shardSize[nid] ..'\n') 
    end
  end
  if opt.imbalanced==1 and opt.IID==1 then
    -- generated by powerlaw
    if opt.netSize==5 then
      imbalanced_shardSizeFrac={0.02359516, 0.11179758, 0.2, 0.28820242, 0.37640484}
    end
    if opt.netSize==10 then
      imbalanced_shardSizeFrac={0.01179758, 0.03139812, 0.05099865, 0.07059919, 0.09019973,0.10980027, 0.12940081, 0.14900135, 0.16860188, 0.18820242}
    end
    if opt.netSize==20 then
      imbalanced_shardSizeFrac={0.00589879, 0.01054102, 0.01518325, 0.01982549, 0.02446772,0.02910995, 0.03375219, 0.03839442, 0.04303665, 0.04767888,
       0.05232112, 0.05696335, 0.06160558, 0.06624781, 0.07089005,
       0.07553228, 0.08017451, 0.08481675, 0.08945898, 0.09410121}
    end
    print('imbalanced shardSizeFrac for total '.. opt.netSize .. ' parties:')
    -- generated by uniform distribution
    -- imbalanced_shardSizeFrac={}
    -- for i=1,opt.netSize do 
    --   imbalanced_shardSizeFrac[i]=torch.uniform(0.1, 0.9)
    -- end
    -- local frac_sum=0
    -- for i = 1, opt.netSize do
    --     frac_sum = frac_sum + imbalanced_shardSizeFrac[i]
    -- end
    -- for i=1,opt.netSize do 
    --   imbalanced_shardSizeFrac[i]=imbalanced_shardSizeFrac[i]/frac_sum
    --   print(imbalanced_shardSizeFrac[i]) 
    -- end
    -- balanced: 600 each, total 600*opt.netSize, imbalanced partition among opt.netSize
    total_records=math.ceil(opt.shardSizeFrac * trainSize * opt.netSize)
    local shffl = torch.randperm(trainData:size()-val_size)
    -- local shffl = torch.randperm(#train_shard)
    accessed=1
    for nid = 1, opt.netSize do
      -- imbalanced_shardSize[nid] = math.ceil(imbalanced_shardSizeFrac[nid] * total_records)
      shardSize[nid] = math.ceil(imbalanced_shardSizeFrac[nid] * total_records)
      print('imbalanced shardSize for party '.. nid .. ': ' .. shardSize[nid] ..'\n')
      trainData.shard[nid] = shffl[{ {accessed,accessed+shardSize[nid]-1} }]
      accessed=accessed+shardSize[nid]
      -- local shffl = torch.randperm(trainData:size()-val_size)
      -- trainData.shard[nid] = shffl[{ {1,shardSize[nid]} }]
    end
  end
  -- non IID data shard 
  if opt.IID==0 then
    train_class_order_file=paths.concat(opt.folder, 'train_class_order.' .. opt.shardID .. '.' .. opt.run) 
    train_class_num_file=paths.concat(opt.folder, 'train_class_num.' .. opt.shardID .. '.' .. opt.run) 
    print('non-IID data shard')
    local class_len=10
    class_indices={}
    for i=1,class_len do
      -- jth raw record corresponds to indices th item with label i
      class_indices[i]={}
      local indices=0
      for j=1,trainSize do
        if trainData.labels[j]==i then
          indices=indices+1
          class_indices[i][indices]=j
        end
      end
    end
    -- print(class_indices)
    -- print(type(class_indices))
    -- print(type(class_indices[1]))
    for nid = 1, opt.netSize do
      -- trainData.shard[nid]={}
      local l=0
      shardSize[nid] = math.ceil(opt.shardSizeFrac * trainSize)
      trainData.shard[nid]=torch.FloatTensor(shardSize[nid]):zero()
      biased_class = torch.random(class_len)
      print(biased_class)
      biased_class_len = math.floor(shardSize[nid]*0.5)
      left_classes_len = math.floor(shardSize[nid]*0.5)
      shuff = torch.randperm(#class_indices[biased_class])
      -- trainData.shard[nid] = class_indices[biased_class][{ shuff[{ {1,biased_class_len} }] }]
      for s=1,biased_class_len do
        l=l+1
        -- print(shuff[s])
        trainData.shard[nid][l] = class_indices[biased_class][shuff[s]]
        -- trainData.shard[nid][l] = class_indices[biased_class][{ shuff[{ {s} }] }]
      end
      left_classes_Frac={}
      for i=1,class_len do 
        if i==biased_class then
          left_classes_Frac[i]=0
        else
          left_classes_Frac[i]=torch.uniform(0.1, 0.9)
        end
      end
      local frac_sum=0
      for i = 1, class_len do
          frac_sum = frac_sum + left_classes_Frac[i]
      end
      if biased_class==class_len then
        last_class=class_len-1
      else
        last_class=class_len
      end
      left_class_len={}
      per_class_num={}
      local classes_len=0
      for i=1,class_len do 
        if i==biased_class then
          left_class_len[i]=0
          per_class_num[i]=biased_class_len
        else
          -- ensure each party has shardSize[nid] examples, by allocating more examples to the last left class
          if i==last_class then
            left_class_len[i]=left_classes_len-classes_len
          else
            -- print(math.floor(left_classes_Frac[i]/frac_sum*left_classes_len))
            left_class_len[i]=math.floor(left_classes_Frac[i]/frac_sum*left_classes_len)
            classes_len=classes_len+left_class_len[i]
          end
          per_class_num[i]=left_class_len[i]
          -- c=torch.FloatTensor(left_class_len[i]):zero()
          -- trainData.shard[nid]=torch.cat(trainData.shard[nid],c)
          shuff = torch.randperm(#class_indices[i])
          for s=1,left_class_len[i] do
            l=l+1
            -- print(shuff[s])
            trainData.shard[nid][l] = class_indices[i][shuff[s]]
          end
          -- l=l+1
          -- trainData.shard[nid][l] = class_indices[i][{ shuff[{ {1,left_class_len[i]} }] }]
        end
      end
      -- per_class_num: sort as per ascending order, improve generalisation after collaboration: per-class test acc vs class distribution
      train_class_num, train_class_order = torch.FloatTensor(per_class_num):abs():sort(1)
    end
    -- userdata
    -- print(type(trainData.shard[1]))
    print(trainData.shard[1])
    print(trainData.shard[1][1])
    torch.save(train_class_order_file, train_class_order)
    torch.save(train_class_num_file, train_class_num)
  end
  torch.save(shardfile, trainData.shard)
end

print(trainData)
print(testData)

if opt.dataset == 'svhn' then
   -- model parameters
   nfeats = 3
   width  = 32
   height = 32
   classes = {'1','2','3','4','5','6','7','8','9','0'}

elseif opt.dataset == 'mnist' then
   nfeats = 1
   width  = 32
   height = 32
   classes = {'1','2','3','4','5','6','7','8','9','0'}

   if opt.model == 'deep' then
    --    60000
    --     32
    --     32
    -- [torch.LongStorage of size 3]
      trainData.data = trainData.data:squeeze()
      testData.data = testData.data:squeeze()
   end
end

ninputs   = nfeats * width * height
nhiddens  = 128 -- ninputs / 6
nhiddens2 = 64 -- ninputs / 12
noutputs  = 10

nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

-- constructing the model
model = nn.Sequential()

---if paths.filep(modelfile) then
---   model = torch.load(modelfile)
---   print('load model from file')
---else
   if opt.model == 'linear' then
      -- Simple linear model
      model:add(nn.Reshape(ninputs))
      model:add(nn.Linear(ninputs,noutputs))

   elseif opt.model == 'mlp' then
      -- Simple 2-layer neural network, with tanh hidden units
      model:add(nn.Reshape(ninputs))
      model:add(nn.Linear(ninputs,nhiddens))
      model:add(nn.Tanh())
      model:add(nn.Linear(nhiddens,noutputs))

   elseif opt.model == 'deep' then
      -- Deep neural network, with ReLU hidden units
      model:add(nn.Reshape(ninputs))
      model:add(nn.Linear(ninputs,nhiddens))
      model:add(nn.ReLU())
      model:add(nn.Linear(nhiddens,nhiddens2))
      model:add(nn.ReLU())
      model:add(nn.Linear(nhiddens2,noutputs))

      -- model:add(nn.Linear(nstates[3], noutputs))

   elseif opt.model == 'cvn' then
      if opt.dataset == 'mnist' then
         -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
         model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
         model:add(nn.Tanh())
         model:add(nn.SpatialMaxPooling(3, 3, 3, 3))

         -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
         model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
         model:add(nn.Tanh())
         model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

         -- stage 3 : standard 2-layer MLP:
         model:add(nn.Reshape(64*2*2))
         model:add(nn.Linear(64*2*2, 200))
         model:add(nn.Tanh())
         model:add(nn.Linear(200, noutputs))

      elseif opt.dataset == 'svhn' then
         -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
         model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
         model:add(nn.Tanh())
         model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
         model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))
   
         -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
         model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
         model:add(nn.Tanh())
         model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
         model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

         -- stage 3 : standard 2-layer neural network
         model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
         model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
         model:add(nn.Tanh())
         model:add(nn.Linear(nstates[3], noutputs))
      end
   end

   -- define loss
   model:add(nn.LogSoftMax())

-- printing the model
print(model)

criterion = nn.ClassNLLCriterion()

-- prepare for training

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients (1-dim vector)
parameters,gradParameters = model:getParameters()

print(parameters:nElement())

-- same initialization for all hyper-paras setting
if paths.filep(paramfile) then
  parameters:copy(torch.load(paramfile))
  print('load same init parameters from file')
else
  torch.save(paramfile, parameters, 'binary')
end


epoch = 0
print('epoch: ' .. epoch)

function train(e,node,cffl)
   -- epoch tracker
  -- if paths.filep(epochfile) then
  --    epoch = torch.load(epochfile)
  -- else
  --    epoch = 1
  -- end
  -- print('epoch: ' .. epoch)
  --  epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- train standalone and distributed using E=1, B=1, lr=1e-3
  if cffl==0 then
    local_nepochs=1
    -- batchSize=1
    -- learningRate=1e-3
    batchSize=opt.batchSize
    learningRate=0.005
  else 
    local_nepochs=opt.local_nepochs
    batchSize=opt.batchSize
    learningRate = opt.learningRate
  end  
  -- SGD optimizer
  optimState = {
     momentum          = opt.momentum,
     weightDecay       = opt.weightDecay,
     -- learningRate      = opt.learningRate,
     learningRate      = learningRate,
     learningRateDecay = opt.learningRateDecay
  }
  optimMethod = optim.sgd
  for local_epoch = 1, local_nepochs do
    print('local_epoch: '..local_epoch)
   local shuffle = torch.randperm(shardSize[node])

   -- Final loss
   local final_loss

   -- do opt.epochFraction of one epoch
   for t = 1, math.ceil(opt.epochFraction * shardSize[node]), batchSize do
      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t, math.min(t+batchSize-1,shardSize[node]) do
         -- load new sample
         local inx = trainData.shard[node][shuffle[i]]
         local input = trainData.data[inx]
         local target = trainData.labels[inx]

         if opt.type == 'double' then input = input:double() end

         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)

                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          -- inputs[i]:[torch.FloatTensor of size 32x32]
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       final_loss = f

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
    end
  end
   -- accuracy
   accuracy = confusion.mat:diag():sum() / confusion.mat:sum()

   -- time taken
   time = sys.clock() - time
   -- traintime = time * 1000 / trainData.shardsize()
   traintime = time * 1000 / shardSize[node]

   -- next epoch
   confusion:zero()
   -- epoch = epoch + 1

   return accuracy, final_loss
end

function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode
   model:evaluate()
   
   local f = 0

   -- test over test data
   for t = 1,testData:size() do
      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      local err = criterion:forward(pred, target)
      f = f + err
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()

   accuracy = confusion.mat:diag():sum() / confusion.mat:sum()
   testtime = time * 1000

   -- next iteration:
   confusion:zero()

   return accuracy, f/testData:size()
end

function test_perclass()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode
   model:evaluate()
   
   local f = 0
   accuracy={}
   test_class_indices = torch.load(test_classfile, 'binary')
   for i=1,class_len do
      test_inx=test_class_indices[i]
     -- test over per-class test data
     for t = 1,#test_inx do
        -- get new sample
        local input = testData.data[test_inx[t]]
        if opt.type == 'double' then input = input:double() end
        -- testData.labels[t]==i
        local target = testData.labels[test_inx[t]]

        -- test sample
        local pred = model:forward(input)
        local err = criterion:forward(pred, target)
        f = f + err
        confusion:add(pred, target)
     end
     accuracy[i] = confusion.mat:diag():sum() / confusion.mat:sum()
     -- next iteration:
     confusion:zero()
   end
    -- timing
   time = sys.clock() - time
   time = time / testData:size()
   testtime = time * 1000
   return accuracy, f/testData:size()
end

function test_val()
   -- set model to evaluate mode
   model:evaluate()
   
   local f = 0

   -- test over test data
   for t = 1,val_size do
      -- get new sample
      inx=val_shard[t]
      local input = trainData.data[inx]
      if opt.type == 'double' then input = input:double() end
      local target = trainData.labels[inx]

      -- test sample
      local pred = model:forward(input)
      local err = criterion:forward(pred, target)
      f = f + err
      confusion:add(pred, target)
   end

   accuracy = confusion.mat:diag():sum() / confusion.mat:sum()

   -- next iteration:
   confusion:zero()
   
   return accuracy, f/val_size
end