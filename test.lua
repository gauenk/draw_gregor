require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'image'

rnn_size = 100

h_dec_prev = nn.Identity()()
gx_tilda = nn.Linear(rnn_size,5)(h_dec_prev)
encoder = nn.gModule({h_dec_prev}, {gx_tilda})
graph.dot(encoder.fg,'encoder')


   for i = 1,N do
      mu_xi = nn.CAddTable()({gx,nn.MulConstant(tmp1_A[i])(delta)})
      mu_yi = nn.CAddTable()({gy,nn.MulConstant(tmp1_B[i])(delta)})
      for a = 1,A do
	 fillA = nn.AddConstant(-a)(mu_xi)
	 fillA = nn.Power(2)(fillA)
	 fillA = nn.CDivTable()({fillA,sigma_sq})
	 nonorm_Fx[#nonorm_Fx+1] = nn.Exp()(fillA)
      end
      for b = 1,B do
	 fillB = nn.AddConstant(-b)(mu_yi)
	 fillB = nn.Power(2)(fillB)
	 fillB = nn.CDivTable()({fillB,sigma_sq})
	 nonorm_Fy[#nonorm_Fy+1] = nn.Exp()(fillB)
      end
      filterA[#filterA+1] = nn.JoinTable(2)(nonorm_Fx)
      filterB[#filterB+1] = nn.JoinTable(2)(nonorm_Fy)

   end
   filterA = nn.JoinTable(2)(filterA)
   filterB = nn.JoinTable(2)(filterB)
