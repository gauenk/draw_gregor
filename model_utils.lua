require 'torch'

local model_utils = {}

function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end




function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function model_utils.clone_many_times_params(net, T, savename, epoch, needGrad, splitLossX)
    local clones = {}

    if count_params == nil then
       count_params = 0
    end
    if count_gparams == nil then
       count_gparams = 0
    end


    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local load_params
    if splitLossX then
       print('loading splitLossX')
       print(splitLossX)
       load_params = splitLossX:parameters()
       params=load_params
    else
       load_params = torch.load(paths.concat(savename[1] .. '/' .. savename[2] .. '_params_' .. epoch .. '.t7'))
       

       -- store loaded parameters in actual parameters
       for i = 1, #params do
	  for j = 1, (params[i]):size(1) do
	     if (params[i]):size(1) ~= 1 and (params[i]):dim() == 2 then
		for k = 1, (params[i][j]):size(1) do
		   count_params = count_params + 1
		   params[i][j][k] = load_params[count_params]
		end
	     elseif (params[i]):size(1) == 1 and (params[i]):dim() == 2 then
		for k = 1, (params[i]):size(2) do
		   count_params = count_params + 1
		   params[i][j][k] = load_params[count_params]
		end
	     else
		count_params = count_params + 1
		params[i][j] = load_params[count_params]
	     end
	  end
       end
    end
    
    if needGrad then
       local load_gradParams = torch.load(paths.concat(savename[1] .. '/' .. savename[2] .. '_gradParams_' .. epoch .. '.t7'))

       for i = 1, #gradParams do
	  for j = 1, (gradParams[i]):size(1) do
	     if (gradParams[i]):size(1) ~= 1 and (gradParams[i]):dim() == 2 then
		for k = 1, (gradParams[i][j]):size(1) do
		   count_gparams = count_gparams + 1
		   gradParams[i][j][k] = load_gradParams[count_gparams]
		end
	     elseif (gradParams[i]):size(1) == 1 and (gradParams[i]):dim() == 2 then
		for k = 1, (gradParams[i]):size(2) do
		   count_gparams = count_gparams + 1
		   gradParams[i][j][k] = load_gradParams[count_gparams]
		end
	     else
		count_gparams = count_gparams + 1
		gradParams[i][j] = load_gradParams[count_gparams]
	     end
	  end
       end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
       paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
	       cloneParams[i]:set(params[i])
	       cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

return model_utils
