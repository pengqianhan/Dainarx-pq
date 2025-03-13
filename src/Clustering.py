from src.CurveSlice import Slice


def clustering(data: list[Slice], self_loop=False):
    tot_mode = 1
    last_mode = None
    mode_dict = {}
    for i in range(len(data)):
        if data[i].isFront:
            last_mode = None

        if Slice.Method == 'dis':
            for j in range(i):
                if (data[j].mode != last_mode) and (data[i] & data[j]):
                    data[i].setMode(data[j].mode)
                if data[i].mode is not None:
                    break
        else:
            for idx, val in mode_dict.items():
                if idx == last_mode:
                    continue
                if data[i].test_set(val[0], val[1], val[2]):
                    data[i].mode = idx
                    if len(mode_dict[idx][0]) < 3:
                        mode_dict[idx][0].append(data[i].data)
                        mode_dict[idx][1].append(data[i].input_data)
                        mode_dict[idx][2] = max(mode_dict[idx][2], data[i].fit_dim)
                    break

        if data[i].mode is None:
            data[i].mode = tot_mode
            mode_dict[tot_mode] = [[data[i].data], [data[i].input_data], data[i].fit_dim]
            tot_mode += 1
        if not self_loop:
            last_mode = data[i].mode
