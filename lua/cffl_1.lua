dofile "cffl_sgd.lua"
require 'csvigo'
require 'image'
require 'paths'
----------------------------------------------------------------
if opt.imbalanced==1 then
  if opt.plevel==1 then
    file = io.open("logs/cffl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_plevel0"..opt.plevel.."_uploadlevel"..opt.uploadFraction.."_imbalanced_IID".. opt.IID .. "_pretrain".. opt.pretrain.. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_alpha"..opt.alpha  .. '_dssgd' ..opt.method .. "_" .. opt.run ..".log", "w")
  else
    file = io.open("logs/cffl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_plevel0"..opt.plevel.."_imbalanced_IID".. opt.IID .. "_pretrain".. opt.pretrain.. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_alpha"..opt.alpha .. '_dssgd' ..opt.method .. "_" .. opt.run ..".log", "w")
  end
else
  if opt.plevel==1 then
    file = io.open("logs/cffl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_plevel0"..opt.plevel.."_uploadlevel" ..opt.uploadFraction.."_IID" .. opt.IID .. "_pretrain".. opt.pretrain.. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_alpha"..opt.alpha  .. '_dssgd' ..opt.method .. "_" .. opt.run ..".log", "w")
  else
    file = io.open("logs/cffl_"..opt.dataset.."_"..opt.model.."_p"..opt.netSize.."e"..opt.nepochs.."_plevel0"..opt.plevel.."_IID".. opt.IID .. "_pretrain".. opt.pretrain.. "_localepoch".. opt.local_nepochs .. "_localbatch".. opt.batchSize .. "_lr".. opt. learningRate .. "_alpha"..opt.alpha .. '_dssgd' ..opt.method .. "_" .. opt.run ..".log", "w")
  end
end

-- sets the default output file as file
io.output(file)
----------------------------------------------------------------
function normalize(t)
   local z = 0.0
   for j = 1, opt.netSize do
      z = z + t[j]
   end
   for j = 1, opt.netSize do
      t[j] = t[j] / z
   end
   return t
end

----------------------------------------------------------------
function fileExists(path)
    local file = io.open(path, "r")
    if file then
        io.close(file)
        return true
    end
    return false
end
----------------------------------------------------------------
local nid
local train_acc = {}
local test_acc = {}
local privacy_level = {}
local credit = {}
local sacc = {}
local ps = {} -- parameters of all nodes for standalone
local ps_10 = {} -- each node train 10 epochs before collaboration
local pserver = parameters * 0.0
pserver:copy(parameters)
local pserver_dssgd = parameters * 0.0
pserver_dssgd:copy(parameters)

if paths.filep(psfile) and paths.filep(ps10file) then
  ps = torch.load(psfile, 'binary')
  ps_10 = torch.load(ps10file, 'binary')
else
  -- same initialization for all parties to begin collaboration, or for standalone training
   for nid = 1, opt.netSize do
      ps[nid] = parameters * 0.0
      ps[nid]:copy(parameters)
      ps_10[nid] = parameters * 0.0
      ps_10[nid]:copy(parameters)
   end
end

for nid = 1, opt.netSize do
  -- same plevel for each party
  if opt.plevel==1 then
      privacy_level[nid] = opt.uploadFraction
  else 
    if paths.filep(plevelfile) then
      print('load plevel')
      privacy_level = torch.load(plevelfile, 'binary')
    else
      print('allocate plevel')
      privacy_level[nid] = torch.uniform(0.1, 1) --magic no. should be controled by hyper-p
    end
  end 
end

if opt.plevel==5 and not paths.filep(plevelfile) then
  torch.save(plevelfile,privacy_level, 'binary')
end
----------------------------------------------------------------
-- pre-train aprior model
for nid = 1, opt.netSize do
  sacc[nid] = 0
  train_acc[nid] = 0
  test_acc[nid] = 0
end
-- each party evaluate other parties using its trained standalone model
-- standalone initial acc, already quite high
if opt.IID==0 then
  sorted_sacc={}
  sorted_max_sacc={}
  for c=1,class_len do
    sorted_max_sacc[c]=0
  end
end
max_acc_standalone={}
for i=1,opt.nepochs do 
  max_acc_standalone[i] = 0
end
for nid = 1, opt.netSize do
  if paths.filep(psfile) then
    print("\nload pre-trained aprior models")
    parameters:copy(ps[nid])
  else
    print("\nPre-train aprior models")
    cffl=0
    parameters:copy(ps[nid])
    for e = epoch+1,opt.nepochs do
      accuracy, loss = train(e,nid,cffl)
      accuracy, loss = test()
      io.write('standalone epoch ' .. e .. ' party ' .. nid .. ' test acc ' .. accuracy,'\n')
      max_acc_standalone[e]=math.max(max_acc_standalone[e],accuracy)
      if e==opt.pretrain_epochs then
        ps_10[nid]:copy(parameters)
      end
    end
    -- standalone model
    ps[nid]:copy(parameters) 
  end
  if opt.IID==1 then
    accuracy, loss = test()
  else
    accuracy, loss = test_perclass()
  end
  sacc[nid] = accuracy
  -- print(p[nid])
  if opt.IID==1 then
    io.write('standalone party ' .. nid .. ' test acc ' .. sacc[nid],'\n')
  else
    print('standalone party ' .. nid .. ' test acc ','\n')
    print(torch.Tensor(sacc[nid]))
    sorted_sacc[nid]=torch.sort(torch.Tensor(sacc[nid]))
  end
end
for e=1,opt.nepochs do 
  io.write('in epoch '.. e .. ', max standalone test acc '.. max_acc_standalone[e],'\n')
end

-- ps at opt.nepochs
if not paths.filep(psfile) or not paths.filep(ps10file) then
  torch.save(psfile, ps, 'binary')
  torch.save(ps10file, ps_10, 'binary')
end
if opt.IID==0 then
  for c=1,class_len do
    for nid = 1, opt.netSize do
      -- X：10 classes (from min to max number of records)
      -- Y: max acc each party can achieve, consistent with IID case 
      sorted_max_sacc[c]=math.max(sorted_max_sacc[c],sorted_sacc[nid][c])
    end
  end
  print('standalone party max test acc as per ascending order of class distribution:')
  print(torch.Tensor(sorted_max_sacc))
end

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end
-----------------------------------------------------------------
for i = 1, opt.netSize do
  -- credit[i] = 1/opt.netSize
  credit[i] = 0
  io.write('\nparty ' .. i .. ' initial credit given by server: ' .. credit[i],'\n')
end
-----------------------------------------------------------------
if opt.credit_thres==1 then
  credit_threshold = 1/opt.netSize*2/3
  decay=0.999
else
  credit_threshold = 0
end

----------------------------------------------------------------
-- continue from the epoch ckpt
local g = {} -- gradeints of all nodes
local g_o = {} -- gradeints of all nodes
local g_dssgd = {} -- gradeints of all nodes
for nid = 1, opt.netSize do
  g_dssgd[nid] = parameters * 0.0
  g[nid] = parameters * 0.0
  g_o[nid] = parameters * 0.0
end

-- local p_dssgd = {} -- parameters of all nodes for collaboration
-- for nid = 1, opt.netSize do
--   p_dssgd[nid] = parameters * 0.0
--   -- p_dssgd[nid]:copy(parameters)
--   io.write('dssgd, load same init parameters','\n')
-- end
local nup_dssgd = math.ceil(opt.uploadFraction * parameters:nElement())

local p = {} -- parameters of all nodes for collaboration
if paths.filep(pfile) then
   p = torch.load(pfile, 'binary')
else
  if opt.pretrain==1 then
   for nid = 1, opt.netSize do
      p[nid] = parameters * 0.0 
      p[nid]:copy(ps_10[nid])
      io.write('load pre-trained paras of 10 epochs','\n')
   end
  else
    for nid = 1, opt.netSize do
      p[nid] = parameters * 0.0
      p[nid]:copy(parameters)
      io.write('load same init parameters','\n')
    end
  end
end

nupdates = parameters * 0.0 -- statistics about parameters' update at the server
max_acc=0
for e = epoch+1,opt.nepochs do
  -- standlone=0
  cffl=1
  if opt.credit_thres==1 then 
    credit_threshold=credit_threshold*decay
  end
  io.write('credit threshold in epoch ' .. e .. ' ' .. credit_threshold,'\n')
  if opt.IID==1 then
    max_acc_dppdl=0
  else
    max_acc_dppdl={}
    for c=1,class_len do
      max_acc_dppdl[c]=0
    end
  end

  -- in each epoch: server integrates all updates into previous paras
  local combined_p = parameters * 0.0
  combined_p:copy(pserver)
  local combined_g = parameters * 0.0
  local upload = {}
  for i = 1, opt.netSize do
    parameters:copy(p[i])
    accuracy, loss = train(e,i,cffl)
    train_acc[i]=accuracy
    io.write('train\tepoch ' .. e .. '\tparty ' .. i .. '\t acc ' .. train_acc[i],'\n')
    if opt.IID==1 then
      accuracy, loss = test()
    else
      accuracy, loss = test_perclass()
    end
    test_acc[i] = accuracy
    if opt.IID==1 then
      io.write('test\tepoch ' .. e .. '\tparty ' .. i .. '\t acc ' .. test_acc[i],'\n')
    else
      print('test\tepoch ' .. e .. '\tparty ' .. i .. '\t acc ','\n')
      print(torch.Tensor(test_acc[i]),'\n')
    end
    -- calc gradient and update the parameters
    g[i]:copy(parameters - p[i])
    if opt.range > 0 then
      -- impose the bound, avoid gradient exploding
      g[i]:apply(function(x) return cut(x, opt.range) end)
    end
    upload[i] = parameters * 0.0
    if privacy_level[i]==1 then
      io.write('interagte all party '.. i ..' updates','\n')
      -- upload[i]=g[i] / opt.netSize
      upload[i]=g[i]
      -- upload[i]=g[i]*credit[i]
      combined_p = combined_p + upload[i]
      combined_g = combined_g + upload[i]
    else
      local upload_num = math.floor(parameters:nElement() * privacy_level[i])
      if upload_num > 0 then         
        local delta = parameters * 0.0
        local temp_delta = parameters * 0.0
        delta:copy(g[i])
        if opt.update_criteria=='large' then
          temp_delta:copy(delta)
          -- largest values gradients
          _, inx = temp_delta:abs():sort(1)
          threshold = temp_delta[ inx[-upload_num] ]
         for elmnt = 1, upload_num do
            inx_ = inx[{ elmnt - upload_num - 1 }]
            upload[i][inx_]=delta[inx_]/ 1.0
            combined_p[inx_] = combined_p[inx_]+upload[i][inx_]
            -- combined_p[inx_] = combined_p[inx_] + (upload[i][inx_] / opt.netSize)
            combined_g[inx_] = combined_g[inx_]+upload[i][inx_]
            nupdates[inx_] = nupdates[inx_] + 1
         end
        else
          -- random threshold
          threshold=0.0001
          local perm = torch.randperm(parameters:nElement())
          local count = 0
          upval = 0

          for elmnt = 1, parameters:nElement() do
            local eid = perm[elmnt] 
            if math.abs(delta[eid]) >= threshold then
              upval = delta[eid]
              combined_p[eid] = combined_p[eid] + (upval / 1.0) 
              -- average sgd
              -- combined_p[eid] = combined_p[eid] + (upval / opt.netSize)
              combined_g[eid] = combined_g[eid] + (upval / 1.0) 
              count = count + 1
              if count >= upload_num then
                break
              end
              upload[i][eid]=upval
              nupdates[eid] = nupdates[eid] + 1
            end
          end
        end  
        io.write('upload gradient threshold ' .. threshold,'\n')
      end
    end  
    ------------------------------------    
  end
  if opt.IID==1 then
    for i = 1, opt.netSize do
      max_acc_dppdl=math.max(max_acc_dppdl,test_acc[i])
    end
    io.write('in epoch '.. e .. ', max dppdl test acc '.. max_acc_dppdl,'\n')
    max_acc=math.max(max_acc,max_acc_dppdl)
  else
    for c=1,class_len do
      for i = 1, opt.netSize do
        max_acc_dppdl[c]=math.max(max_acc_dppdl[c],test_acc[i][c])
      end
      io.write('in epoch '.. e .. ', max dppdl test acc for class '.. c-1 ..': '..max_acc_dppdl[c],'\n')
      max_acc[c]=math.max(max_acc[c],max_acc_dppdl[c])
    end
  end
  pserver:copy(combined_p)
  parameters:copy(pserver)
  accuracy, loss = test()
  io.write('in epoch '.. e .. ', server test acc '.. accuracy,'\n')

  -- evaluate individual contribution (val acc) using each party's local model, only applicable to upload_rate=1 and pretrain=0, if upload_rate=0.1, server cannot know the whole p[i], only 10% of p[i], then use parameters:copy(pserver+upload[i]) 
  val_acc={}
  sum=0
  credit_epoch={}
  for i = 1, opt.netSize do
    -- pretrain=0 and upload_rate=1
    if opt.uploadFraction==1 and opt.pretrain==0 then
      -- privacy_level[nid] = 1
      io.write('use p[i] model to compute val_acc','\n')
      parameters:copy(p[i]+upload[i])
    -- pretrain=1 or upload_rate~=1
    else
      io.write('use pserver to compute val_acc','\n')
      parameters:copy(pserver+upload[i])
    end
    val_acc[i], loss = test_val()
    io.write('in epoch '.. e .. ', party '..i.. ' val acc '.. val_acc[i],'\n')
    sum=sum+val_acc[i]  
  end 
  for i = 1, opt.netSize do 
    credit_epoch[i]=val_acc[i]/sum
    if opt.credit_fade==1 then
      credit[i] = credit[i]*0.2+credit_epoch[i]*0.8
    else
      credit[i] = (credit[i]+credit_epoch[i])/2
    end
    if credit[i] < credit_threshold then
      credit[i]=0
    end
    -- sinh(alpha): alpha punishment factor, current results use sinh(5) for uploadFraction=1 and uploadFraction=0.1, tuned alpha could be better
    credit[i]=math.sinh(opt.alpha*credit[i])
  end

  -- -- normalize credit
  credit = normalize(credit)
  io.write('server updates normalized credit','\n')

  for i = 1, opt.netSize do
    io.write('\nafter epoch '.. e .. ', party ' .. i .. ' credit given by server:' .. credit[i],'\n')
  end
  -- local download = {}
  -- download para from server
  for i = 1, opt.netSize do    
   -- for i, download the credit[i]*#parameters combined grads from server according to the “largest values” criterion; This can be speedup in parallel
    if credit[i] >= credit_threshold then
      local ndown = math.floor(credit[i] * parameters:nElement())
      -- local ndown = math.floor(credit[i]/math.max(table.unpack(credit)) * parameters:nElement() * 0.2)
      -- local ndown = math.floor(credit[i]/math.max(table.unpack(credit)) * parameters:nElement())
      -- local ndown = math.floor(credit[i]/math.max(table.unpack(credit)) * parameters:nElement() * 0.5) --the party with highest credit can download whole/half global model to ensure accuracy     
      io.write('party ' .. i .. ' ndown ' .. ndown,'\n')
      local delta = parameters * 0.0
      local temp_delta = parameters * 0.0
      delta:copy(combined_g)
      if opt.update_criteria=='large' then
        temp_delta:copy(delta)
        -- largest values gradients
        _, inx = temp_delta:abs():sort(1)
        for elmnt = 1, ndown do
          inx_ = inx[{ elmnt - ndown - 1 }]
          -- download allocated combined_g, excluding its own corresponding upload[i]
          p[i][inx_] = p[i][inx_]+delta[inx_]-upload[i][inx_]
        end
        p[i] = p[i]+g[i] --each party combines its own gradients, and combined_g excluding its own upload[i]
      end
    end 
  end

  -- -- dssgd+synchronous: each party sends 10% updates after opt.local_nepochs=1 (local_nepochs=5 converge very slow), and downloads the same whole global paras
  -- if opt.method=='syn' then
  --   cffl=0 --opt.local_nepochs=1
  --   -- cffl=1 --opt.local_nepochs=5
  --   for i = 1, opt.netSize do
  --     parameters:copy(pserver_dssgd)
  --     accuracy, loss = train(e,i,cffl)
  --     -- save the parameters and gradient of node i
  --     -- p_dssgd[i]:copy(parameters)
  --     g_dssgd[i]:copy(parameters - pserver_dssgd)
  --     if opt.range > 0 then
  --        -- impose the bound, and determine the sensitivity
  --      g_dssgd[i]:apply(function(x) return cut(x, opt.range) end)
  --     end
  --     local delta = parameters * 0.0
  --     local temp_delta = parameters * 0.0
  --     delta:copy(g_dssgd[i])        
  --     if opt.update_criteria=='large' then
  --       temp_delta:copy(delta)
  --       -- largest values gradients
  --       _, inx = temp_delta:abs():sort(1)
  --       threshold = temp_delta[ inx[-nup_dssgd] ]
  --      for elmnt = 1, nup_dssgd do
  --         inx_ = inx[{ elmnt - nup_dssgd - 1 }]
  --         pserver_dssgd[inx_] = pserver_dssgd[inx_]+delta[inx_]/ 1.0
  --      end
  --     end
  --   end
  --   parameters:copy(pserver_dssgd) 
  --   accuracy = test()
  --   io.write('in epoch '.. e .. ', dssgd server test acc '.. accuracy,'\n')
  -- end

  -- dssgd+seq
  if opt.method=='seq' then
    cffl=0 --opt.local_nepochs=1
    -- cffl=1 --opt.local_nepochs=5
    for i = 1, opt.netSize do
      parameters:copy(pserver_dssgd)
      accuracy = test()
      io.write('after download pserver_dssgd, in epoch '.. e .. ', dssgd party ' .. i.. ' test acc '.. accuracy,'\n')
      accuracy, loss = train(e,i,cffl)
      -- save the parameters and gradient of node i
      -- p_dssgd[i]:copy(parameters)
      g_dssgd[i]:copy(parameters - pserver_dssgd)
      if opt.range > 0 then
         -- impose the bound, and determine the sensitivity
       g_dssgd[i]:apply(function(x) return cut(x, opt.range) end)
      end
      local delta = parameters * 0.0
      local temp_delta = parameters * 0.0
      delta:copy(g_dssgd[i])        
      if opt.update_criteria=='large' then
        temp_delta:copy(delta)
        -- largest values gradients
        _, inx = temp_delta:abs():sort(1)
        threshold = temp_delta[ inx[-nup_dssgd] ]
       for elmnt = 1, nup_dssgd do
          inx_ = inx[{ elmnt - nup_dssgd - 1 }]
          pserver_dssgd[inx_] = pserver_dssgd[inx_]+delta[inx_]/ 1.0
       end
      end
      accuracy = test()
      io.write('in epoch '.. e .. ', dssgd party ' .. i.. ' test acc '.. accuracy,'\n')
    end
  end
end
----------------------------------------------------------------
-- fairness computation
for i = 1, opt.netSize do
  io.write("party " .. i .. " shard_size: " .. shardSize[i],'\n')
  io.write("party " .. i .. " privacy level: " .. privacy_level[i],'\n')
  io.write('party ' .. i .. ' final credit ' .. credit[i],'\n')
  if opt.IID==1 then
    io.write('party ' .. i .. ' final test acc ' .. test_acc[i],'\n')
  else
    print('party ' .. i .. ' final test acc ')
    print(torch.Tensor(test_acc[i]))
    print('party ' .. i .. ' final mean test acc ')
    print(torch.mean(torch.Tensor(test_acc[i])))
  end
end
if opt.IID==1 then
  io.write('in all epochs ' .. ', max dppdl test acc '.. max_acc)
else
  -- cannot io.write and .. table
  print('in all epochs ' .. ', max dppdl test acc ')
  print(torch.Tensor(max_acc))
end
-- closes the open file
io.close(file)