function read(hdp,const_matA,const_matB,x,x_error)
   local gx,gy, sigma_sq, delta,i_vec,j_vec,mu_x,mu_y,Fx_unnormal,Fy_unnormal,Fx,Fy,gx_tilda, gy_tilda,log_sigma_sq, log_delta, log_gamma, gamma
   gx_tilda = nn.Linear(rnn_size,1)(hdp)
   gy_tilda = nn.Linear(rnn_size,1)(hdp)
   log_sigma_sq = nn.Linear(rnn_size,1)(hdp)
   log_delta = nn.Linear(rnn_size,1)(hdp)
   log_gamma = nn.Linear(rnn_size,1)(hdp)

   i_vec = torch.linspace(1,N,N)
   j_vec = i_vec:clone()

   -- compute initial variables -- 
   gx = nn.MulConstant((A+1)/2)(nn.AddConstant(1)(gx_tilda))
   gy = nn.MulConstant((B+1)/2)(nn.AddConstant(1)(gy_tilda))
   delta = nn.MulConstant((math.max(A,B)-1)/(N-1))(nn.Exp()(log_delta))
   sigma_sq = nn.Exp()(log_sigma_sq)
   gamma = nn.Squeeze()(nn.Replicate(N,3)(nn.Replicate(2*N,2)(nn.Exp()(log_gamma))))

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
   
   local x_read = nn.MM(false,true)({nn.MM()({filterFx,x}),filterFy})
   local x_error_read = nn.MM(false,true)({nn.MM()({filterFx,x_error}),filterFy})
   
   local concat_output = nn.CMulTable()({nn.JoinTable(2)({x_read,x_error_read}),gamma})

   return concat_output
end
