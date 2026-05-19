import numpy as np

def get_marks(model):

    marks = model.encode('"')[0].tolist()[0]

    return marks

def ft_decoder(input_ids, model):

    raw_logits = model.get_logits_from_input_ids(input_ids)

    mark = get_marks(model)
    idx = -1
    print(mark)
    while int(idx) != mark:
        idx = np.argmax(raw_logits)
        input_ids.append(int(idx))
        raw_logits = model.get_logits_from_input_ids(input_ids)
    # print(input_ids)
    print(model.decode(input_ids))
