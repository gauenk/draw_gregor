require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

seq_length = 30
A = 28
B = 28
N = 12

readAll = torch.load('test_A/readTable_4.t7')
local read_gx,read_gy,read_delta,read_sigmaSq,read_ux,read_uy,mnist_digit
read_gx={}
read_gy={}
read_delta={}
read_sigmaSq={}
read_ux={}
read_uy={}
mnist_digit = {}

print(#readAll)

for t = 1, #readAll do
   read_gx[t],read_gy[t],read_delta[t],read_sigmaSq[t],read_ux[t],read_uy[t],mnist_digit[t] = unpack(readAll[t])
end
--print(read_ux[1][1])
--print(read_uy[1][1])

function x_bound(a,b,upper_bound)
   -- check bounds on rx and lx
   rx = math.floor(a)
   lx = math.floor(b)
   
   if(rx < 1) then
      rx = 1
   end

   if(lx < 1) then
      lx = 1
   end

   if(rx > upper_bound) then
      rx = upper_bound
   end

   if(lx > upper_bound) then
      lx = upper_bound
   end
   
   if(rx - lx < .001) then
--      print("Error: x bounds are equal!")
   end
   
   return rx,lx
end

function add_mu_boxes(t,i,y)
   local ux = read_ux[t][i]:select(2,1)
   local uy = read_uy[t][i]:select(2,1)
   print(ux)
   local m,n
   for m = 1, N do
      for n = 1, N do
	 xCor = math.floor(ux[m])
	 yCor = math.floor(uy[n])
	 xCor,yCor = m_bound(math.floor(ux[m]),math.floor(uy[n]),A,B)
	 print(xCor)
	 print(yCor)
	 y[t][2][xCor][yCor] = 255
      end
   end
   
end

function add_read_boxes(t,i,y)
   local ux = read_ux[t][i]:select(2,1)
   local uy = read_uy[t][i]:select(2,1)
   --print(unpack(read_sigmaSq))
   local sigmaSq = read_sigmaSq[t][i]
   --   print(sigmaSq)
   local rl = torch.max(ux)
   local rl = torch.max(ux)


   local rx = torch.max(ux)
   local lx = torch.min(ux)

   local ty = torch.max(uy)
   local by = torch.min(uy)

   --print(string.format('before -- r: %d | l: %d | t: %d | b: %d',rx,lx,ty,by))
   rx, lx = x_bound(rx,lx,A) -- checks bounds
   ty, by = x_bound(ty,by,B) -- checks bounds

   for k = lx, rx do
      y[t][2][k][by] = 255
   end

   for k = lx, rx do
      y[t][2][k][ty] = 255
   end

   for k = by, ty do
      y[t][2][lx][k] = 255
   end

   for k = by, ty do
      y[t][2][rx][k] = 255
   end
   
end


function m_bound(a,b,aBound,bBound)
   local x,y
   x = a
   y = b

   if x < 1 then
      x = 1
   elseif  x > aBound then
      x = aBound
   end

   if y < 1 then
      y = 1
   elseif  y > aBound then
      y = aBound
   end
   
   return x,y
end

function add_boxes(t,i,y)

   local gx = read_gx[t][i][1]
   local gy = read_gy[t][i][1]
   local delta = read_delta[t][i][1]
   
   local rx = gx + delta
   local lx = gx - delta

   local ty = gy + delta
   local by = gy - delta

   --print(string.format('before -- r: %d | l: %d | t: %d | b: %d',rx,lx,ty,by))
   rx, lx = x_bound(rx,lx,A) -- checks bounds
   ty, by = x_bound(ty,by,B) -- checks bounds

   for k = lx, rx do
      y[t][2][k][by] = 255
   end

   for k = lx, rx do
      y[t][2][k][ty] = 255
   end

   for k = by, ty do
      y[t][2][lx][k] = 255
   end

   for k = by, ty do
      y[t][2][rx][k] = 255
   end

end

counter = 1
for i = 1, mnist_digit[1]:size(1) do --number in "n_data".. default = 20
   c = torch.zeros(#mnist_digit, 3, mnist_digit[1]:size(2), mnist_digit[1]:size(3)) --3 for RGB
   
   for t = 1, #mnist_digit do --number in sequence... default = 50
      for k = 1,mnist_digit[1]:size(2) do
	 for m = 1,mnist_digit[1]:size(2) do
	    c[t][1][k][m] = mnist_digit[t][i][k][m]*255
	    c[t][2][k][m] = mnist_digit[t][i][k][m]*255
	    c[t][3][k][m] = mnist_digit[t][i][k][m]*255
	 end
      end
      add_mu_boxes(t,i,c)
   end
   print(counter)
   counter = counter + 1
   --  seq_length = 1
   --  for t = 1,seq_length do
   --     diff = image.drawText(c[t], "hello\nworld", 10, 0,{color = {255, 0, 0}, bg = {0, 255, 255}, size = 10,inplace=true}) - c[t]
   --     print(torch.sum(diff))
   --  end

   --  image.display(image.toDisplayTensor(image.drawText(c[1], "hello\nworld", 3, 5,{color = {255, 0, 0}, bg = {0, 255, 255}, size = 10,inplace=true})))
   image.display({image = (c)})

end

