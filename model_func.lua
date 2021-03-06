function lstm(input,prev_output,prev_cell,curr_size)
   
   function new_input_sum()
      local i2h = nn.Linear(curr_size,curr_size)(input)
      local h2h = nn.Linear(curr_size,curr_size)(prev_output)
      return nn.CAddTable()({i2h,h2h}):annotate{
	 name = 'new_input_sum'}
   end

   local input_gate = nn.Sigmoid()(new_input_sum()):annotate{
      name = 'lstm: input gate'}
   local output_gate = nn.Sigmoid()(new_input_sum()):annotate{
      name = 'lstm: output gate'}
   local forget_gate = nn.Sigmoid()(new_input_sum()):annotate{
      name = 'lstm: forget gate'}
   local input_transform = nn.Tanh()(new_input_sum()):annotate{
      name = 'lstm: input transform'}

   local next_cell = nn.CAddTable()({
	 nn.CMulTable()({forget_gate,prev_cell}),
	 nn.CMulTable()({input_gate,input_transform}):annotate{
	    name = 'next lstm cell value'}
				   })
   local next_output = nn.CMulTable()({output_gate,nn.Tanh()(next_cell)}):annotate{
      name='encoder lstm output',
      graphAttributes = {style='filled',fillcolor='yellow'}
    }
   
   return next_output,next_cell
end

function write(h_dec,const_matA,const_matB,w_t)
   local gx,gy, sigma_sq, delta,i_vec,j_vec,mu_x,mu_y,Fx_unnormal,Fy_unnormal,Fx,Fy,gx_tilda, gy_tilda,log_sigma_sq, log_delta, log_gamma, gamma
   gx_tilda = nn.Linear(decoder_size,1)(h_dec):annotate{
      name = 'gx_tilda'}
   gy_tilda = nn.Linear(decoder_size,1)(h_dec):annotate{
      name = 'gy_tilda'}
   log_sigma_sq = nn.Linear(decoder_size,1)(h_dec):annotate{
      name = 'log_sigma_sq'}
   log_delta = nn.Linear(decoder_size,1)(h_dec):annotate{
      name = 'log_delta'}
   log_gamma = nn.Linear(decoder_size,1)(h_dec):annotate{
      name = 'log_gamma'}

   i_vec = torch.linspace(1,N,N)
   j_vec = i_vec:clone()

   -- compute initial variables -- 
   gx = nn.MulConstant((A+1)/2)(nn.AddConstant(1)(gx_tilda))
   gy = nn.MulConstant((B+1)/2)(nn.AddConstant(1)(gy_tilda))
   delta = nn.MulConstant((math.max(A,B)-1)/(N-1))(nn.Exp()(log_delta))
   sigma_sq = nn.Exp()(log_sigma_sq)
   gamma = nn.Squeeze()(nn.Replicate(B,3)(nn.Replicate(A,2)(nn.Exp()(log_gamma))))
   inv_gamma = nn.Power(-1)(gamma)
   
   -- filterbanks -- 
   local const_A = i_vec:add(-N/2-1/2)
   local const_B = j_vec:add(-N/2-1/2)
   local mu_xi = {}
   local mu_yi = {}
   
   for i = 1,N do
      mu_xi[#mu_xi+1] = nn.CAddTable()({gx,nn.MulConstant(const_A[i])(delta)})
      mu_yi[#mu_yi+1] = nn.CAddTable()({gy,nn.MulConstant(const_B[i])(delta)})
   end

   local mu_x = nn.Replicate(A,3,N)(nn.JoinTable(2)(mu_xi)):annotate{
      name = 'mu_x Joining+Rep'}

   local mu_y = nn.Replicate(B,3,N)(nn.JoinTable(2)(mu_yi)):annotate{
      name = 'mu_y Joining+Rep'}

   inv_sigmasq_A = nn.Squeeze()(nn.Replicate(A,3)(nn.Replicate(N,2)(nn.Power(-1)(nn.MulConstant(2)(sigma_sq)))))
   inv_sigmasq_B = nn.Squeeze()(nn.Replicate(B,3)(nn.Replicate(N,2)(nn.Power(-1)(nn.MulConstant(2)(sigma_sq)))))

   local numeratorA = nn.Power(2)(nn.CSubTable()({mu_x,const_matA}))
   local numeratorB = nn.Power(2)(nn.CSubTable()({mu_y,const_matB}))
   numeratorA = nn.MulConstant(-1)(nn.CMulTable()({inv_sigmasq_A,numeratorA}))
   numeratorB = nn.MulConstant(-1)(nn.CMulTable()({inv_sigmasq_B,numeratorB}))

   local unnormal_Fx = nn.Exp()(numeratorA)
   local unnormal_Fy = nn.Exp()(numeratorB)

   local sumX = nn.Squeeze()((nn.Replicate(N,2)(nn.Replicate(A,2)(nn.Sum(2)(nn.Sum(2)(unnormal_Fx)))))):annotate{
      name = 'sum X'}
   
   local sumY = nn.Squeeze()(nn.Replicate(N,2)(nn.Replicate(B,2)(nn.Sum(2)(nn.Sum(2)(unnormal_Fy))))):annotate{
      name = 'sum Y'}
   
   local filterFx = nn.CDivTable()({unnormal_Fx,sumX}):annotate{
      name = 'normalizing Fx'}
   local filterFy = nn.CDivTable()({unnormal_Fy,sumY}):annotate{
      name = 'normalizing Fy'}
   
   local w_write = nn.MM()({nn.MM(true,false)({filterFx,w_t}),filterFy})
   
   local concat_output = nn.CMulTable()({w_write,inv_gamma}):annotate{
      name='reshape write output',graphAttributes = {style='filled',fillcolor='yellow'}}

   return concat_output,gx,gy,delta,sigma_sq
end

function generate_encoder(arg)
   -- input variables
   x = nn.Identity()():annotate{
      name = 'x-- input image'}

   lstm_h_dec_prev = nn.Identity()():annotate{
      name = 'lstm_h_dec_prev'}

   sig_canvas_prev = nn.Identity()():annotate{
      name = 'sig_canvas_prev'}

   local const_matA = nn.Identity()():annotate{
      name = 'ConstantMat A'}

   local const_matB = nn.Identity()():annotate{
      name = 'ConstantMat B'}

   epsilon = nn.Identity()():annotate{
      name = 'epsilon: normal noise'}

   -- read --
   x_error = nn.CSubTable()({x,sig_canvas_prev}):annotate{
      name = 'x_error'}

   local concat_output,gx,gy,delta,sigmaSq,ux,uy = read(lstm_h_dec_prev,const_matA,const_matB,x,x_error)

   -- LSTM 
   lstm_c_enc_prev = nn.Identity()():annotate{
      name = 'lstm_c_enc_prev'}
   lstm_h_enc_prev = nn.Identity()():annotate{
      name = 'lstm_h_enc_prev'}
   h_enc,next_lstm_c_enc = lstm(nn.JoinTable(2)({concat_output,lstm_h_dec_prev}),lstm_h_enc_prev,lstm_c_enc_prev,encoder_size)

   mu_t = nn.Linear(encoder_size,n_z)(h_enc):annotate{
      name = 'mu of z_t'}
   sigma_t = nn.Exp()(nn.Linear(encoder_size,n_z)(h_enc)):annotate{
      name= 'sigma of z_t'}

   local z = nn.CAddTable()({nn.CMulTable()({sigma_t,epsilon}),mu_t}):annotate{
      name = 'sample of z'}

   local loss_z = nn.AddConstant(-1)(nn.Sum(2)(nn.CAddTable()({ nn.Square()(mu_t), nn.Square()(sigma_t), nn.MulConstant(-1)(nn.Log()(nn.Square()(sigma_t)))  }))):annotate{
      name = 'loss_z'}
   
   local encoder
   if arg then
      if arg == 'attention' then
	 encoder = nn.gModule({x,sig_canvas_prev,lstm_h_dec_prev,const_matA,const_matB,lstm_c_enc_prev,lstm_h_enc_prev,epsilon}, {z,loss_z,h_enc,next_lstm_c_enc,concat_output,gx,gy,delta,sigmaSq,ux,uy})
	 encoder.name = 'attention_encoder'
	 print(string.format('attention encoder\n'))
      else
	 encoder = nn.gModule({x,sig_canvas_prev,lstm_h_dec_prev,const_matA,const_matB,lstm_c_enc_prev,lstm_h_enc_prev,epsilon}, {z,loss_z,h_enc,next_lstm_c_enc,concat_output})
	 encoder.name = 'encoder'
      end
   else
      encoder = nn.gModule({x,sig_canvas_prev,lstm_h_dec_prev,const_matA,const_matB,lstm_c_enc_prev,lstm_h_enc_prev,epsilon}, {z,loss_z,h_enc,next_lstm_c_enc,concat_output})
      encoder.name = 'encoder'
   end

   return encoder
end

function generate_decoder(arg)
   z_input = nn.Identity()():annotate{
      name = 'z_input'}

   h_dec_prev = nn.Identity()():annotate{
      name = 'lstm_h_enc_prev'}

   lstm_cell_prev = nn.Identity()():annotate{
      name = 'lstm_c_enc_prev'}

   local const_matA = nn.Identity()():annotate{
      name = 'ConstantMat A'}

   local const_matB = nn.Identity()():annotate{
      name = 'ConstantMat B'}

   prev_canvas = nn.Identity()():annotate{
      name = 'Previous Canvas'}

   target_value = nn.Identity()():annotate{
      name = 'Taget Value'}

   h_dec,lstm_c_next = lstm(z_input,h_dec_prev,lstm_cell_prev,decoder_size)
   w_t = nn.Reshape(N,N,true)(nn.Linear(decoder_size,N*N)(h_dec)):annotate{
      name='w_t'}

   write_output,gx,gy,delta,sigma_sq = write(h_dec,const_matA,const_matB,w_t)
   
   curr_canvas = nn.CAddTable()({write_output,prev_canvas})

   sig_canvas = nn.Sigmoid()(curr_canvas)
   
   --loss_x = nn.MulConstant(-1)(nn.Sum(2)(nn.Sum(2)(nn.CAddTable()({nn.CMulTable()({nn.AddConstant(1)( nn.MulConstant(-1)(target_value) ), nn.Log()(nn.AddConstant(1)( nn.MulConstant(-1)(target_value))) }),nn.CMulTable()({target_value,nn.Log()(sig_canvas)})}))))
   
   loss_x_pt1 = nn.CMulTable()({target_value,nn.Log()(sig_canvas)})
   loss_x_pt2A = nn.AddConstant(1)(nn.MulConstant(-1)(target_value))
   loss_x_pt2B = nn.Log()(nn.AddConstant(1)(nn.MulConstant(-1)(sig_canvas)))
   loss_x_pt2 = nn.CMulTable()({loss_x_pt2A,loss_x_pt2B})
   loss_x = nn.MulConstant(-1)(nn.Sum(2)(nn.Sum(2)(nn.CAddTable()({loss_x_pt1,loss_x_pt2}))))

   local decoder
   if arg then
      decoder = nn.gModule({z_input,h_dec_prev,lstm_cell_prev,const_matA,const_matB,prev_canvas,target_value},{h_dec,lstm_c_next,curr_canvas,sig_canvas,loss_x,write_output,gx,gy,delta,sigma_sq,loss_x_pt1,nn.Log()(sig_canvas)})
      decoder.name = 'splitLoss_decoder'
      print('decoder:splitLoss')
   else
      decoder = nn.gModule({z_input,h_dec_prev,lstm_cell_prev,const_matA,const_matB,prev_canvas,target_value},{h_dec,lstm_c_next,curr_canvas,sig_canvas,loss_x,write_output,gx,gy,delta,sigma_sq})
      decoder.name = 'decoder'

   end

   return decoder
end

function read(hdp,const_matA,const_matB,x,x_error)
   local gx,gy, sigma_sq, delta,i_vec,j_vec,mu_x,mu_y,Fx_unnormal,Fy_unnormal,Fx,Fy,gx_tilda, gy_tilda,log_sigma_sq, log_delta, log_gamma, gamma
   gx_tilda = nn.Linear(decoder_size,1)(hdp):annotate{
      name = 'gx_tilda'}
   gy_tilda = nn.Linear(decoder_size,1)(hdp):annotate{
      name = 'gy_tilda'}
   log_sigma_sq = nn.Linear(decoder_size,1)(hdp):annotate{
      name = 'log_sigma_sq'}
   log_delta = nn.Linear(decoder_size,1)(hdp):annotate{
      name = 'log_delta'}
   log_gamma = nn.Linear(decoder_size,1)(hdp):annotate{
      name = 'log_gamma'}

   -- compute initial variables -- 
   gx = nn.MulConstant((A+1)/2)(nn.AddConstant(1)(gx_tilda)):annotate{
      name = 'gx'}
   gy = nn.MulConstant((B+1)/2)(nn.AddConstant(1)(gy_tilda)):annotate{
      name = 'gy'}
   delta = nn.MulConstant((math.max(A,B)-1)/(N-1))(nn.Exp()(log_delta)):annotate{
      name = 'delta'}
   sigma_sq = nn.Exp()(log_sigma_sq):annotate{
      name = 'sigma_sq'}
   gamma = nn.Squeeze()(nn.Replicate(N,3)(nn.Replicate(2*N,2)(nn.Exp()(log_gamma)))):annotate{
      name = 'gamma'}

   -- filterbanks -- 
   local i_vec = torch.linspace(1,N,N)
   local j_vec = i_vec:clone()
   local const_A = i_vec:add(-N/2-1/2)
   local const_B = j_vec:add(-N/2-1/2)
   local mu_xi = {}
   local mu_yi = {}
   
   for i = 1,N do
      mu_xi[#mu_xi+1] = nn.CAddTable()({gx,nn.MulConstant(const_A[i])(delta)}):annotate{
      name = 'mu_xi'}
      mu_yi[#mu_yi+1] = nn.CAddTable()({gy,nn.MulConstant(const_B[i])(delta)}):annotate{
      name = 'mu_yi'}
   end

   local mu_x = nn.Replicate(A,3,N)(nn.JoinTable(2)(mu_xi)):annotate{
      name = 'mu_x Joining+Rep'}

   local mu_y = nn.Replicate(B,3,N)(nn.JoinTable(2)(mu_yi)):annotate{
      name = 'mu_y Joining+Rep'}

   inv_sigmasq_A = nn.Squeeze()(nn.Replicate(A,3)(nn.Replicate(N,2)(nn.Power(-1)(nn.MulConstant(2)(sigma_sq))))):annotate{
      name = 'inv_sigmasq_A'}
   inv_sigmasq_B = nn.Squeeze()(nn.Replicate(B,3)(nn.Replicate(N,2)(nn.Power(-1)(nn.MulConstant(2)(sigma_sq))))):annotate{
      name = 'inv_sigmasq_B'}

   local numeratorA = nn.Power(2)(nn.CSubTable()({mu_x,const_matA})):annotate{
      name = 'numeratorA pt1'}
   local numeratorB = nn.Power(2)(nn.CSubTable()({mu_y,const_matB})):annotate{
      name = 'numeratorB pt1'}
   numeratorA = nn.MulConstant(-1)(nn.CMulTable()({inv_sigmasq_A,numeratorA})):annotate{
      name = 'numeratorA pt2'}
   numeratorB = nn.MulConstant(-1)(nn.CMulTable()({inv_sigmasq_B,numeratorB})):annotate{
      name = 'numeratorB pt2'}

   local unnormal_Fx = nn.Exp()(numeratorA):annotate{
      name = 'unnormal_Fx'}
   local unnormal_Fy = nn.Exp()(numeratorB):annotate{
      name = 'unnormal_Fy'}

   local sumX = nn.Squeeze()((nn.Replicate(N,2)(nn.Replicate(A,2)(nn.Sum(2)(nn.Sum(2)(unnormal_Fx)))))):annotate{
      name = 'sum X'}
   
   local sumY = nn.Squeeze()(nn.Replicate(N,2)(nn.Replicate(B,2)(nn.Sum(2)(nn.Sum(2)(unnormal_Fy))))):annotate{
      name = 'sum Y'}
   
   local filterFx = nn.CDivTable()({unnormal_Fx,sumX}):annotate{
      name = 'normalizing Fx'}
   local filterFy = nn.CDivTable()({unnormal_Fy,sumY}):annotate{
      name = 'normalizing Fy'}
   
   local x_read = nn.MM(false,true)({nn.MM()({filterFx,x}),filterFy}):annotate{
      name = 'x_read'}
   local x_error_read = nn.MM(false,true)({nn.MM()({filterFx,x_error}),filterFy}):annotate{
      name = 'x_error_read'}
   
   local concat_output = nn.Reshape(batch_size,2*N*N)(nn.CMulTable()({nn.JoinTable(2)({x_read,x_error_read}),gamma})):annotate{
      name='reshape read output',graphAttributes = {style='filled',fillcolor='yellow'}}

   return concat_output,gx,gy,delta,sigma_sq,mu_x,mu_y
end

function feval(params_)
   --------------------------------------
   --------- FEVAL FUNCTION  ------------
   --------------------------------------
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()
    
    --------------- MNIST MINI-BATCH -----------------
    local start_index = counter * batch_size + 1
    local end_index = math.min(n_train_data, (counter + 1) * batch_size)
    if end_index == n_train_data then
       counter = 0
    else
       counter = counter + 1
    end

    local batch_inputs = train_data[{{start_index, end_index}, {}}]
    local batch_targets = train_label[{{start_index, end_index}}]
    --local curr_batch_size = end_index - start_index + 1
    --batch_inputs = torch.reshape(batch_inputs,curr_batch_size,28*28)
    
    nidx = (nidx or 0) + 1
    if nidx > (#train_data)[1] then nidx = 1 end

    ------------------- FORWARD PASS -------------------
    local loss = 0
    local z = {}
    local x = {}
    local loss_z = {}
    local lstm_c_enc = {[0]=torch.zeros(batch_size, encoder_size)}
    local lstm_h_enc = {[0]=torch.zeros(batch_size, encoder_size)}
    local lstm_c_dec = {[0]=torch.zeros(batch_size, decoder_size)}
    local lstm_h_dec = {[0]=torch.zeros(batch_size, decoder_size)}
    local x_error = {[0]=torch.rand(batch_size, A, B)}
    local x_prediction = {}
    local w = {}
    local read_section = {}
    local write_section = {}
    local loss_x = {}
    local canvas = {[0]=torch.rand(batch_size, A, B)}
    local sig_canvas = {[0]=torch.rand(batch_size, A, B)}
    local rand_input = {}
    local dec_gx = {}
    local dec_gy = {}
    local dec_delta = {}
    local dec_sigmaSq = {}
    local enc_gx = {}
    local enc_gy = {}
    local enc_delta = {}
    local enc_sigmaSq = {}
    local enc_ux = {}
    local enc_uy = {}
    local loss_x_pt1
    local loss_x_pt2

    for t = 1, time_steps do
       x[t] = batch_inputs
       rand_input[t] = torch.randn(batch_size,n_z)
       z[t], loss_z[t], lstm_h_enc[t], lstm_c_enc[t],read_section[t],enc_gx[t],enc_gy[t],enc_delta[t],enc_sigmaSq[t],enc_ux,enc_uy = unpack(encoder_clones[t]:forward({x[t],sig_canvas[t-1], lstm_h_dec[t-1], const_matA, const_matB, lstm_c_enc[t-1], lstm_h_enc[t-1],rand_input[t]}))
       lstm_h_dec[t],lstm_c_dec[t],canvas[t],sig_canvas[t],loss_x[t],write_section[t],dec_gx[t],dec_gy[t],dec_delta[t],dec_sigmaSq[t] = unpack(decoder_clones[t]:forward({z[t],lstm_h_dec[t-1],lstm_c_dec[t-1],const_matA,const_matB,canvas[t-1],x[t]}))
       loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
       if isnan(loss) == true then
	  isnan_error(t,{z[t],lstm_h_dec[t-1],lstm_c_dec[t-1],const_matA,const_matB,canvas[t-1],x[t]})
       end
    end
    loss = loss / time_steps
    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_c_enc = {[time_steps] = torch.zeros(batch_size, encoder_size)}
    local dlstm_h_enc = {[time_steps] = torch.zeros(batch_size, encoder_size)}
    local dlstm_c_dec = {[time_steps] = torch.zeros(batch_size, decoder_size)}
    local dlstm_h_dec = {[time_steps] = torch.zeros(batch_size, decoder_size)}
    local dlstm_h_dec1 = {[time_steps] = torch.zeros(batch_size, decoder_size)}
    local dlstm_h_dec2 = {[time_steps] = torch.zeros(batch_size, decoder_size)}
    local dx_error = {[time_steps] = torch.zeros(batch_size, A, B)}
    local dcanvas = {[time_steps] = torch.zeros(batch_size, A, B)}
    local dsig_canvas = {[time_steps] = torch.zeros(batch_size, A, B)}
    local drand_input = {[time_steps] = torch.zeros(batch_size, n_z)}
    local dz = {[time_steps] = torch.zeros(batch_size, A, B)}

    local dread_section = {}
    local dwrite_section = {}
    local dloss_x_pt1 = {}
    local dloss_x_pt2 = {}
    local ddec_gx = {}
    local ddec_gy = {}
    local ddec_delta = {}
    local ddec_sigmaSq = {}
    local denc_gx = {}
    local denc_gy = {}
    local denc_delta = {}
    local denc_sigmaSq = {}
    local denc_ux = {}
    local denc_uy = {}
    local dloss_z = {}
    local dloss_x = {}
    local dx1 = {}
    local dx2 = {}
    local dx = {}
    for t = time_steps,1,-1 do
       dloss_x[t] = torch.ones(batch_size, 1)
       dloss_z[t] = torch.ones(batch_size, 1)
       dread_section[t] = torch.zeros(batch_size, 2*N*N)
       dwrite_section[t] = torch.zeros(batch_size, A, B)
       ddec_gx[t] = torch.zeros(batch_size,1)
       ddec_gy[t] = torch.zeros(batch_size,1)
       ddec_delta[t] = torch.zeros(batch_size,1)
       ddec_sigmaSq[t] = torch.zeros(batch_size,1)
       denc_gx[t] = torch.zeros(batch_size,1)
       denc_gy[t] = torch.zeros(batch_size,1)
       denc_delta[t] = torch.zeros(batch_size,1)
       denc_sigmaSq[t] = torch.zeros(batch_size,1)
       denc_ux[t] = torch.zeros(batch_size,1)
       denc_uy[t] = torch.zeros(batch_size,1)

       --       z[t], loss_z[t], lstm_h_enc[t], lstm_c_enc[t],read_section[t] = unpack(encoder_clones[t]:forward({x[t],sig_canvas[t-1], lstm_h_dec[t-1], const_matA, const_matB, lstm_c_enc[t-1], lstm_h_enc[t-1],rand_input[t]}))
       --       lstm_h_dec[t],lstm_c_dec[t],canvas[t],sig_canvas[t],loss_x[t],write_section[t],dec_gx[t],dec_gy[t],dec_delta[t],dec_sigmaSq[t] = unpack(decoder_clones[t]:forward({z[t],lstm_h_dec[t-1],lstm_c_dec[t-1],const_matA,const_matB,canvas[t-1],x[t]}))
       dz[t], dlstm_h_dec1[t-1], dlstm_c_dec[t-1], dconst_matA, dconst_matB, dcanvas[t-1],dx1[t] = unpack(decoder_clones[t]:backward({z[t],lstm_h_dec[t-1],lstm_c_dec[t-1],const_matA,const_matB,canvas[t-1],x[t]}, {dlstm_h_dec[t],dlstm_c_dec[t],dcanvas[t],dsig_canvas[t],dloss_x[t],dwrite_section[t],ddec_gx[t],ddec_gy[t],ddec_delta[t],ddec_sigmaSq[t]}))
       --use if including attention variables in output
--       dx2[t], dsig_canvas[t-1], dlstm_h_dec2[t-1], dconst_matA, dconst_matB, dlstm_c_enc[t-1], dlstm_h_enc[t-1],drand_input[t] = unpack(encoder_clones[t]:backward({x[t],sig_canvas[t-1], lstm_h_dec[t-1], const_matA, const_matB, lstm_c_enc[t-1], lstm_h_enc[t-1],rand_input[t]}, {dz[t], dloss_z[t],dlstm_h_enc[t], dlstm_c_enc[t],dread_section[t],denc_gx[t],denc_gy[t],denc_delta[t],denc_sigmaSq[t],denc_ux[t],denc_uy[t]}))
dx2[t], dsig_canvas[t-1], dlstm_h_dec2[t-1], dconst_matA, dconst_matB, dlstm_c_enc[t-1], dlstm_h_enc[t-1],drand_input[t] = unpack(encoder_clones[t]:backward({x[t],sig_canvas[t-1], lstm_h_dec[t-1], const_matA, const_matB, lstm_c_enc[t-1], lstm_h_enc[t-1],rand_input[t]}, {dz[t], dloss_z[t],dlstm_h_enc[t], dlstm_c_enc[t],dread_section[t]}))
       dlstm_h_dec[t-1] = dlstm_h_dec1[t-1] + dlstm_h_dec2[t-1]
       dx[t] = dx1[t] + dx2[t]
    end

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end


function isnan_error(t,input)
   local model_utils = require 'model_utils'
   --------------------------------------
   ---------- MAKING MODEL --------------
   --------------------------------------
   splitLoss_decoder = generate_decoder('errorSplit')

   -- make T clones
   print('---- starting: cloning ----')
   count_params = nil
   splitLoss_decoder_clones = model_utils.clone_many_times_params(splitLoss_decoder,time_steps, savename, retrain_epoch,false,decoder)

   print('---- finished: cloning ----')

   _,_,_,sig_canv,_,_,_,_,_,_,loss_x_pt1,log_sig_canv = unpack(splitLoss_decoder_clones[t]:forward({input[1],input[2],input[3],input[4],input[5],input[6],input[7]}))
   print(torch.sum(loss_x_pt1))
   print(torch.sum(log_sig_canv))
   print(torch.sum(sig_canv))
   torch.save(paths.concat(savename[1] .. '/' .. savename[1] .. '_nanErr_params_' .. current_epoch .. '.t7'), params)
   torch.save(paths.concat(savename[1] .. '/' .. savename[1] .. '_nanErr_gradParams_' .. current_epoch .. '.t7'), grad_params)
--   torch.save(paths.concat(savename[1] .. '/' .. savename[1] .. '_nanErr_params_' .. current_iteration .. '_' .. current_epoch .. '.t7'), params)
--   torch.save(paths.concat(savename[1] .. '/' .. savename[1] .. '_nanErr_gradParams_' .. current_iteration .. '_' .. current_epoch .. '.t7'), grad_params)

   print(torch.sum(input[7]))
   -- kills process
   print(loss[t])

end

function test_isnan_error()
   isnan_error(1,{torch.zeros(batch_size,n_z),torch.zeros(batch_size,decoder_size),torch.zeros(batch_size,decoder_size),torch.zeros(batch_size,N,A),torch.zeros(batch_size,N,B),torch.zeros(batch_size,A,B),torch.zeros(batch_size,A,B)})
end

function get_read_params(input)
   local model_utils=require 'model_utils'
   local attEncoder_clones,attEncoder
   
   attEncoder = generate_encoder('attention')
   -- flatter parameters
   count_params = nil

--   attEncoder_clones = model_utils.clone_many_times_params(attEncoder,time_steps, savename, retrain_epoch,false,encoder)

   attEncoder_clones = model_utils.clone_many_times(attEncoder,time_steps)

   local loss = 0
   local z = {}
   local x = {}
   local loss_z = {}
   local lstm_c_enc = {[0]=torch.zeros(batch_size, encoder_size)}
   local lstm_h_enc = {[0]=torch.zeros(batch_size, encoder_size)}
   local lstm_c_dec = {[0]=torch.zeros(batch_size, decoder_size)}
   local lstm_h_dec = {[0]=torch.zeros(batch_size, decoder_size)}
   local x_error = {[0]=torch.rand(batch_size, A, B)}
   local x_prediction = {}
   local w = {}
   local read_section = {}
   local write_section = {}
   local loss_x = {}
   local canvas = {[0]=torch.rand(batch_size, A, B)}
   local sig_canvas = {[0]=torch.rand(batch_size, A, B)}
   local rand_input = {}
   local dec_gx = {}
   local dec_gy = {}
   local dec_delta = {}
   local dec_sigmaSq = {}
   local enc_gx = {}
   local enc_gy = {}
   local enc_delta = {}
   local enc_sigmaSq = {}
   local enc_ux = {}
   local enc_uy = {}
   local batch_inputs = train_data[{{1, batch_size}, {}}]
   local read_table = {}

   for t = 1, time_steps do

      rand_input[t] = torch.randn(batch_size,n_z)

       z[t], loss_z[t], lstm_h_enc[t], lstm_c_enc[t],read_section[t],enc_gx[t],enc_gy[t],enc_delta[t],enc_sigmaSq[t],enc_ux[t],enc_uy[t] = unpack(encoder_clones[t]:forward({input[1],input[2],input[3],input[4],input[5],input[6],input[7],rand_input[t]}))

      z[t], loss_z[t], lstm_h_enc[t], lstm_c_enc[t],read_section[t],enc_gx[t],enc_gy[t],enc_delta[t],enc_sigmaSq[t],enc_ux[t],enc_uy[t] = unpack(attEncoder_clones[t]:forward({input[1],input[2],input[3],input[4],input[5],input[6],input[7],rand_input[t]}))

      read_table[t] = {enc_gx[t],enc_gy[t],enc_delta[t],enc_sigmaSq[t],enc_ux[t],enc_uy[t],input[1]}
   end
   print(read_table)
   torch.save(paths.concat(savename[1] .. '/readTable_' .. retrain_epoch .. '.t7'),read_table)

end


function get_error_iteration()
   local output = io.popen('ls ' .. savename[1] .. ' | grep _nanErr_params | awk -F_ \'{print $4}\' | awk -F. \'{print $1}\'')
   
   local result = output:read("*a")
   -- filter to max epoch
   

   local stringVals = split(result,'\n')
   local rTensor = torch.Tensor(string_num(stringVals))
   local maxVal = rTensor:max()
   print(string.format("max = %d",rTensor:max()))
   return maxVal
end

function string_num(itable)
   local t = {}
   for i = 1, #itable do
      t[i] = tonumber(itable[i])
   end

   return t
end

function split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
	 table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end
