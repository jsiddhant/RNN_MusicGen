def sample_lstm(dictionary, sample_len = 100, model_path='models/lstm.pt', start_char = '<start>', temperature = 1):
    
    out_list = []
    # Load Model
    model = LSTM(len(dictionary), 150, len(dictionary), computing_device, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.reset_state()
    
    #Init Starting Sample
    current_sequence_input = torch.zeros(1, 1, len(dictionary))
    current_sequence_input [0,0,dictionary.class_to_index[start_char]]=1
    current_sequence_input = current_sequence_input.to(computing_device)
    
    out_list.append(start_char)
    
    for i in range(sample_len):
        
        output = model(current_sequence_input)
        probabilities = nn.functional.softmax(output.div(temperature), dim=2)
        prob = probabilities.data.cpu().numpy().flatten().astype('float64')
        prob = prob / prob.sum()
        out_ind = np.argmax(np.random.multinomial(1, prob, size=1)[0])
#         out_list.append(dictionary.index_to_class[torch.argmax(probabilities.data)])
        out_list.append(dictionary.index_to_class[out_ind])

        current_sequence_input = torch.zeros(1, 1, len(dictionary))
        current_sequence_input [0,0,dictionary.class_to_index[start_char]]=1
        current_sequence_input = current_sequence_input.to(computing_device)
    
    return out_list
    
