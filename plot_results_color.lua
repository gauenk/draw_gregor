require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

seq_length = 30
A = 28
B = 28
--x_prediction = torch.load('x_generation')
--boxes = torch.load('boxes')

--x_prediction = torch.load('test_A/write_canvas_1.t7')
--boxes = torch.load('test_A/write_writeTable_1.t7')
x_prediction = torch.load('modelB/canvas_3.t7')
boxes = torch.load('modelB/writeTable_3.t7')


x = torch.zeros(#x_prediction, x_prediction[1]:size(2), x_prediction[1]:size(3)) 
y = torch.zeros(#x_prediction, 3, x_prediction[1]:size(2), x_prediction[1]:size(3))

local model = nn.Sequential()
model:add(nn.Sigmoid())

--x_prediction[1] = model:forward(x_prediction[1]):apply(function(x)
--      if x < 0 then
--	 return 0
--      else
--	 return x
--      end
--end)


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

function add_read_boxes(t,i,y)
   local ux = read_ux[t][i]:select(2,1)
   local uy = read_uy[t][i]:select(2,1)
   print(unpack(read_sigmaSq))
   local sigmaSq = read_sigmaSq[t][i]
   print(sigmaSq)
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

function add_boxes(t,i,y)

   local gx = boxes[t][1][i][1]
   local gy = boxes[t][2][i][1]
   local delta = boxes[t][3][i][1]

   local rx = gx + delta
   local lx = gx - delta

   local ty = gy + delta
   local by = gy - delta

   print(string.format('before -- r: %d | l: %d | t: %d | b: %d',rx,lx,ty,by))
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
for i = 1, x_prediction[1]:size(1) do --number in "n_data".. default = 20
   c = torch.zeros(#x_prediction, 3, x_prediction[1]:size(2), x_prediction[1]:size(3)) --3 for RGB

  for t = 1, #x_prediction do --number in sequence... default = 50
    x[{{t}, {}, {}}] = model:forward(x_prediction[t][i])
    
    for k = 1,x_prediction[1]:size(2) do
       for m = 1,x_prediction[1]:size(2) do
	  c[t][1][k][m] = x[t][k][m]*255
	  c[t][2][k][m] = x[t][k][m]*255
	  c[t][3][k][m] = x[t][k][m]*255
       end
    end

    add_boxes(t,i,c)
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

--BOX STRUCTURE
-- 50x3x20x28
-- 50 for sequence length
-- 3 for number of variables: gx = idx 1, gy = idx2, delta = idx3
-- 20 for batch size ~ 20 different images
-- 28 for vector~ used in previous code for filterbank arithmetic

--for i = 1, boxes[1][1]:size(1) do
--  for t = 1, #boxes do

     -- so now we have a vector of size 28 (for ea. row and column) 
     -- we want to iterate over the "n_data" examples
     -- we want to iterate over the whole sequence length
     -- we have a vector, but only need one of the values... let's take the 1st

     -- update idea: only save the value, not the whole vector. no reason to store so much...
     
--     local gx = boxes[t][1][i][1]
--     local gy = boxes[t][2][i][1]
--     local delta = boxes[t][3][i][1]

--     local rx = gx + delta
--     local lx = gx - delta
     
--     rx, lx = x_bound(rx,lx) -- checks bounds

--     local ty = gy + delta
--     local by = gy - delta

--     ty, by = x_bound(ty,by) -- checks bounds

--     for k = lx, rx do
--	y[t][k][by][2] = 255
--     end

--     for k = lx, rx do
--	y[t][k][ty][2] = 255
--     end

--     for k = by, ty do
--	y[t][lx][k][2] = 255
--     end

--     for k = by, ty do
--	y[t][rx][k][2] = 255
--     end

--  end

--  image.display(y)
--  y = torch.zeros(#x_prediction, x_prediction[1]:size(2), x_prediction[1]:size(3))

--end

