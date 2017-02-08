function forward_pass()
   local loss = 0
   local const_matA = torch.zeros(batch_size,N,A)
   local const_matB = torch.zeros(batch_size,N,B)
   local z = {}
   local loss_z = {}
   local lstm_c_enc = {[0]=torch.zeros(batch_size, encoder_size)}
   local lstm_h_enc = {[0]=torch.zeros(batch_size, encoder_size)}
   local lstm_c_dec = {[0]=torch.zeros(batch_size, decoder_size)}
   local lstm_h_dec = {[0]=torch.zeros(batch_size, decoder_size)}
   local x_error = {[0]=torch.rand(batch_size, A, B)}
   local x_prediction = {}
   local w = {}
   local loss_x = {}
   local canvas = {[0]=torch.rand(batch_size, A, B)}
   local inputs = torch.rand(batch_size,A,B)
   local sig_canvas = {[0]=torch.rand(batch_size, A, B)}

   for t = 1, time_steps do
      x[t] = inputs
      z[t], loss_z[t], lstm_h_enc[t], lstm_c_enc[t] = unpack(encoder_clones[t]:forward({x[t],sig_canvas[t-1], lstm_h_dec[t-1], const_matA, const_matB, lstm_c_enc[t-1], lstm_h_enc[t-1],torch.randn(batch_size,n_z)}))
      lstm_h_dec[t],lstm_c_dec[t],w[t],canvas[t],sig_canvas[t],loss_x[t] = unpack(decoder_clones[t]:forward({z[t],lstm_h_dec[t-1],lstm_c_dec[t-1],const_matA,const_matB,canvas[t-1],x[t]}))
      loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
   end
   print(torch.mean(loss_x[time_steps]))
   loss = loss / time_steps
   print(loss)
end

function test_encoder()
   print('-=-=-=- TESTING ENCODER -=-=-=-')
   local input = torch.randn(batch_size,A,B)
   local canvas_prev = torch.randn(batch_size,A,B)
   local epsilon = torch.randn(batch_size,decoder_size)
   local initial_prev_cell = torch.zeros(batch_size,encoder_size)
   local init_h_enc_prev = torch.zeros(batch_size,encoder_size)
   local const_matA = torch.zeros(batch_size,N,A)
   local const_matB = torch.zeros(batch_size,N,B)
   local h_dec_prev = torch.ones(batch_size,decoder_size)
   for i = 1,batch_size do
      h_dec_prev[i] = torch.ones(decoder_size)*i
   end

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

   output = encoder:forward({input,canvas_prev,h_dec_prev,const_matA,const_matB,initial_prev_cell,init_h_enc_prev,epsilon})
   graph.dot(encoder.fg,'encoder','test')
   print(output)
   print(output[1])
   print(output[2])
end

function test_decoder()
   print('-=-=-=- TESTING DECODER -=-=-=-')
   local z_input = torch.randn(batch_size,n_z)
   local targets = torch.randn(batch_size,A,B)
   local lstm_cell_prev = torch.zeros(batch_size,decoder_size)
   local const_matA = torch.zeros(batch_size,N,A)
   local const_matB = torch.zeros(batch_size,N,B)
   local h_dec_prev = torch.ones(batch_size,decoder_size)
   local prev_canvas = torch.ones(batch_size,A,B)
   for i = 1,batch_size do
      h_dec_prev[i] = torch.ones(decoder_size)*i
   end

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

   output = decoder:forward({z_input,h_dec_prev,lstm_cell_prev,const_matA,const_matB,prev_canvas,targets})
   graph.dot(decoder.fg,'decoder','test')
   print(output)
   print(output[6])

end

