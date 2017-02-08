require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'model_func'
require 'test_func'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
nngraph.setDebug(true)

epoch_load = 3
-- settings --
N = 12
A = 28
B = 28
encoder_size = 300
decoder_size = encoder_size - 2*N*N
batch_size = 60
n_z = decoder_size -- must be true: see eq 14 from DRAW (v2) Gregor et al
time_steps = 30
n_train_data = 60000
n_test_data = 5000
epoch_count = 5
counter = 0 -- for feval function
retrain_epoch = 3
modelname = 'modelB'
savename = {modelname,modelname}
seeError = 1
if seeError == 1 then
   savename[2] = savename[1] .. '_nanErr'
end

-- adam settings
optimStat = {learningRate = 2e-3,
	     beta1 = .85,
	     beta2 = .999,
	     epsilon = 1e-38
}

-- set the current epoch
if retrain_epoch then
   current_epoch = retrain_epoch + 1
else
   current_epoch = 1
end

--- make model ---
encoder = generate_encoder()
decoder = generate_decoder()

--- load with trained parameters --- 
encoder_clones = model_utils.clone_many_times_params(encoder,time_steps, savename, epoch_load)
decoder_clones = model_utils.clone_many_times_params(decoder,time_steps, savename, epoch_load)

-- make global variable constants
const_matA = torch.zeros(batch_size,N,A)
const_matB = torch.zeros(batch_size,N,B)

for k = 1,batch_size do
   for i = 1,N do
      for a = 1,A do
	 const_matA[k][i][a] = a
      end
      for b = 1,B do
	 const_matB[k][i][b] = b
      end
   end
end

--------------------------------------
-------- PREPARING DATASET -----------
--------------------------------------
print('---- starting: preparing dataset ----')
local trainset = mnist.traindataset()
local testset = mnist.testdataset()

-- shuffle the dataset
local shuffled_indices = torch.randperm(trainset.size):long()

-- creates a shuffled *copy*, with a new storage
train_data = trainset.data:index(1,shuffled_indices):squeeze():apply(function(x)
      if x >= 125 then
	 return 1
      else 
	 return 0
      end
end)
train_label = trainset.label:index(1,shuffled_indices):squeeze()
train_label = train_label:apply(function(x) 
      if x==0 then 
	 x=10 
      end
      return x 
end)
--train_data = train_data:reshape(train_data:size(1), 28*28):type('torch.DoubleTensor')
train_data = train_data:type('torch.DoubleTensor')

--test_data = testset.data:reshape(testset.data:size(1), 28*28):type('torch.DoubleTensor')
test_data = testset.data:apply(function(x)
      if x >= 125 then
	 return 1
      else 
	 return 0
      end
end):type('torch.DoubleTensor')

test_label = testset.label
test_label = test_label:apply(function(x) 
      if x==0 then 
	 x=10 
      end
      return x 
end)

print('---- finished: preparing dataset ----')

----------------------------------------------
-------------- GENERATE SECTION --------------
----------------------------------------------

-- GET READ PARAMETERS
print('--- starting: read parameters ---')
get_read_params({train_data[{{1, batch_size}, {}}],torch.zeros(batch_size,A,B),torch.zeros(batch_size,decoder_size), const_matA, const_matB, torch.zeros(batch_size,encoder_size),torch.zeros(batch_size,encoder_size),torch.randn(batch_size,n_z)})
print('--- finished: read parameters ---')
----------------- GENERATE SECTION -----------------
local z = {}
local canvas = {[0]=torch.zeros(batch_size,A,B)}
local lstm_h_dec = {[0]=torch.zeros(batch_size,decoder_size)}
local lstm_c_dec = {[0]=torch.zeros(batch_size,decoder_size)}
local lstm_h_enc = {[0]=torch.zeros(batch_size,encoder_size)}
local lstm_c_enc = {[0]=torch.zeros(batch_size,encoder_size)}
local sig_canvas = {[0]=torch.zeros(batch_size,A,B)}
local loss_x = {}
local write_section = {}
local batch_inputs = train_data[{{1, batch_size}, {}}]
local random_input = torch.randn(batch_size,n_z)
local dec_gx = {}
local dec_gy = {}
local dec_delta = {}
local dec_sigmaSq = {}
local enc_gx = {}
local enc_gy = {}
local enc_delta = {}
local enc_sigmaSq = {}
local read_section = {}
local loss_z = {}

local write_table = {}
local read_table = {}
print('---- Generating Samples ----')
for t = 1, time_steps do
   z[t] = torch.randn(batch_size,n_z)

   z[t], loss_z[t], lstm_h_enc[t], lstm_c_enc[t],read_section[t],enc_gx[t],enc_gy[t],enc_delta[t],enc_sigmaSq[t] = unpack(encoder_clones[t]:forward({batch_inputs,sig_canvas[t-1], lstm_h_dec[t-1], const_matA, const_matB, lstm_c_enc[t-1], lstm_h_enc[t-1],random_input}))
   lstm_h_dec[t],lstm_c_dec[t],canvas[t],sig_canvas[t],loss_x[t],write_section[t],dec_gx[t],dec_gy[t],dec_delta[t],dec_sigmaSq[t] = unpack(decoder_clones[t]:forward({z[t],lstm_h_dec[t-1],lstm_c_dec[t-1],const_matA,const_matB,canvas[t-1],batch_inputs}))
   write_table[t] = {dec_gx[t],dec_gy[t],dec_delta[t],dec_sigmaSq[t]}
end
print('saving...')
torch.save(paths.concat(savename[1] .. '/canvas_' .. retrain_epoch .. '.t7'),canvas)
torch.save(paths.concat(savename[1] .. '/writeSection_' .. retrain_epoch .. '.t7'),write_section)
torch.save(paths.concat(savename[1] .. '/writeTable_' .. retrain_epoch .. '.t7'),write_table)
print('---- Saved: Canvas + Write Section ----')
----------------- END: GENERATE SECTION -----------------

