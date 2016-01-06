--Make fun with the nngparh
require "nngraph"
require 'torch'
require 'cudnn'
require 'cunn'
require 'base'
require 'sys'
require 'paths'
require 'paths'
require 'nn'
require 'inn'


--nngraph.setDebug(true)
--------------------------------------------
local function debug(vname, vvalue )
  print('-----------'.. vname..'------------')
  print(vvalue)
  print('-----------------------------------')
end
--------------------------------------------

local params = {batch_size=1,
                seq_length=5,
                layers=1,
                decay=2,
                rnn_size=1000,
                dropout=0,
                init_weight=0.1,
                lr=1,
                hidden_size=4096,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5,
                n_output=5}

local function transfer_data(x)
  return x:cuda()
end

local LookupTable = nn.LookupTable


local model = {}
local paramx, paramdx


--------------------An lstm unit------------------------
local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end



-------------------CNN model------------------------
local createModel = paths.dofile('zeiler.lua')
local backend = 'cudnn'
local cnnmodel = createModel(backend)


---------------Create the whole network by nngraph--------------
local function create_network()

  local x                = nn.Identity()()
  
  local cnn_model = cnnmodel
  local cnn_output = cnn_model(x)
  local lstm_input = nn.Identity()(cnn_output)
  local reshaped = nn.Reshape(1,1000)(lstm_input)
  local feat_input = {[0] = reshaped}  --input size 1 * 2048

  --local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
 
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local next_c, next_h = lstm(feat_input[layer_idx - 1], prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    feat_input[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.n_output)
  local dropped          = nn.Dropout(params.dropout)(feat_input[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  --local err              = nn.MSECriterion()({pred, y})
  --For debug 
  --nngraph.annotateNodes()
  local module           = nn.gModule({x,  prev_s},
                                      {pred, nn.Identity()(next_s)})


  module:getParameters():uniform(-params.init_weight, params.init_weight)

  ---------------Just convert the type of the whole module to torch.CudaTensor
  return transfer_data(module)            
end


-------------------------Setup the whole network---------------------------

local function setup()
  print("Creating a CNN-LSTM network...")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()

  ----------------------Init the params to the gpu as the type of torch.CudaTensor----------------------
  -----------------The parameters of the model is to save the states of different steps(seq_length)   

  model.s = {}
  model.ds = {}
  model.start_s = {}
  model.preds = {}
  criterion = nn.MSECriterion()
  for j = 0, params.seq_length do
    model.s[j] = {}
    model.preds[j] = transfer_data(torch.zeros(params.n_output))
    --prev_s contains prev_h and prev_c
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  print("Setting Up Done.")
  --------------------------------------------------------------------------------------------------------
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

-------------------------------feedforward----------------------

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.label[state.pos]
    local s = model.s[i - 1]
    model.preds[i], model.s[i] = unpack(model.rnns[i]:forward({x, s}))
    model.err[i] = criterion:forward(model.preds[i],y)
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.preds,model.err:mean()
end

------------------------------backforward------------------------
local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.label[state.pos]
    local s = model.s[i - 1]
    --local derr = transfer_data(torch.ones(1))
    local gradCriterion = transfer_data(criterion:backward(model.preds[i],y))
    local tmp = model.rnns[i]:backward({x, s},
                                       {gradCriterion, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

------------------------------Debug-------------------------------

torch.manualSeed(1234)

input = torch.rand(5,3,96,96):cuda()

setup()

label = torch.Tensor{{1,1,1,1,0},{2,2,2,2,0.5},{3,3,3,3,1},{4,4,4,4,1.5},{5,5,5,5,2}}:cuda()
train_data = {}
train_data.pos = 1
train_data.data = input
train_data.label = label
data = torch.rand(1,1000):cuda()
preds = {}
local start = torch.tic()
local preds,err = fp(train_data)
local endtime = torch.toc(start)
print(err)
print(string.format('Time using:  %.5f seconds',endtime))
for i = 1 ,#preds do
  print(preds[i])
end