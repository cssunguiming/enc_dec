import pickle
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque

def pad_tensor(vec, pad, dim):
    
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)

    return torch.cat([vec, torch.zeros(*pad_size, dtype=torch.long)], dim=dim)

def before_pad_tensor(vec, pad, dim):

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)

    return torch.cat([torch.zeros(*pad_size, dtype=torch.long), vec], dim=dim)


def generate_input_history(data_neural, mode, candidate=None):
    # train_data, train_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="train")
    data_train = {}
    train_idx = {}

    if candidate is None:
        candidate = list(data_neural.keys())
    for u in candidate:
        sessions = data_neural[u]['sessions']
        sessions_ids = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(sessions_ids):
            trace = {}
            # if mode == 'train' and c == 0:
            #     continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            target_time = np.array([s[1] for s in session[1:]])

            history = []
            if mode == 'test':                                               # train_id = sessions_id[:split_id]
                trained_id = data_neural[u]['train']
                for tt in trained_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[sessions_ids[j]]])

            # history_tim = [t[1] for t in history]
            # history_count = [1]
            # last_t = history_tim[0]
            # count = 1
            # for t in history_tim[1:]:
            #     if t == last_t:
            #         count += 1
            #     else:                           # histort_tim: [1 2 2 2 3 3 4 5 ]
            #         history_count[-1] = count   # history_count: [1 3 2 1 1]
            #         history_count.append(1)
            #         last_t = t
            #         count = 1

            history_loc = np.reshape(np.array([s[0] for s in history]), (-1))
            history_tim = np.reshape(np.array([s[1] for s in history]), (-1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            # trace['history_count'] =-1
            # loc_tim = history
            loc_tim = []
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (-1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (-1))
            his_len_traj = [len(history)]

            trace['loc'] = Variable(torch.LongTensor(loc_np))
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target_loc'] = Variable(torch.LongTensor(target))
            trace['target_tim'] = Variable(torch.LongTensor(target_time))
            trace['his_len_traj'] = Variable(torch.LongTensor(his_len_traj))
            
            data_train[u][i] = trace
        train_idx[u] = sessions_ids
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys())
    train_queue = deque()                                   # train_id = sessions_id[:split_id]
    if mode == 'random':                                    # train_idx = {u:train_id, ....}
        initial_queue = {}
        for u in user:
            if mode2 == 'train':                
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue
