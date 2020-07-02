import torch
import numpy as np
from Tool_Gen_Data import pad_tensor, before_pad_tensor
from Eval import cal_loss_performance, get_acc

def save_model(Epoch, model, file_path="./pretrain/", Predict=False):
    if Predict:
        output_path = file_path + "Predict_model_trained_ep%d.pth" % Epoch
    else:
        output_path = file_path + "Pretrained_ep%d.pth" % Epoch
    # bert_output_path = file_path + "bert_trained_ep%d.pth" % Epoch
    torch.save(model.state_dict(), output_path)
    print("EP:%d Model Saved on:" % Epoch, output_path, "\n")
    return True

def train_epoch(epoch, model, train_data, train_queue, optimizer, device, batch_size):
    # predict_train_epoch(epoch_i, model, train_data, train_queue, optimizer, device)

    model.train()
    desc= ' -(Train)- '
    total_loss, avg_loss, avg_acc = 0, 0, 0.
    iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0
    total_loc, total_cor_loc = 0, 0
    eva_metric = np.zeros((6, 1))

    len_queue = len(train_queue)
    len_batch = int(np.ceil((len_queue/batch_size)))
    train_queue = [[train_queue.popleft() for _ in range(min(batch_size, len(train_queue)))] for k in range(len_batch)]
    
    for i, batch_queue in enumerate(train_queue):

        max_place = max([len(train_data[u][idx]['loc']) for u,idx in batch_queue]) 
        max_history = max([len(train_data[u][idx]['history_loc']) for u,idx in batch_queue]) 
        
        loc = torch.cat([pad_tensor(train_data[u][idx]['loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        time = torch.cat([pad_tensor(train_data[u][idx]['tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        loc_label = torch.cat([pad_tensor(train_data[u][idx]['target_loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
        # time_label = torch.cat([pad_tensor(train_data[u][idx]['target_tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
        
        history_loc = torch.cat([before_pad_tensor(train_data[u][idx]['history_loc'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        history_time = torch.cat([before_pad_tensor(train_data[u][idx]['history_tim'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
        
        # his_len_traj = torch.from_numpy(np.array([train_data[u][idx]['his_len_traj'] for u, idx in batch_queue])).to(device).long()

        # input_loc = torch.cat([history_loc, loc], dim=1)
        # input_tim = torch.cat([history_time, time], dim=1)

        place_logit = model(history_loc, history_time, loc, time)

        loss, n_loc, n_cor_loc, n_cor_time = cal_loss_performance(logit1=place_logit, label1=loc_label, Predict=True)
        eva_metric = get_acc(loc_label, place_logit, eva_metric)

        total_loss += loss.item()
        iter_100_loss += loss.item()
        avg_loss = total_loss/(i+1)

        total_loc += n_loc
        iter_100_loc += n_loc
        total_cor_loc += n_cor_loc
        iter_100_cor_loc += n_cor_loc
        iter_100_cor_time += n_cor_time
        avg_acc = 100.*total_cor_loc/total_loc

        if i % 100 == 0:
            try:
                if n_loc==0 and n_cor_loc==0:
                    n_loc = 1
                print("{} epoch: {:_>2d} | iter: {:_>4d}/{:_>4d} | loss: {:<10.7f} | avg_loss: {:<10.7f} | acc: {:<4.4f} % | avg_acc: {:<4.4f} % | lr: {:<9.7f}".format(
                    desc, epoch, i, len_batch, iter_100_loss/100., avg_loss, 100.*iter_100_cor_loc/iter_100_loc, avg_acc, optimizer._print_lr()))   
                iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0
            except Exception as e:
                print(e)
                exit()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()

    return avg_loss, avg_acc, eva_metric/total_loc

def valid_epoch(epoch, model, valid_data, valid_queue, optimizer, device, batch_size):

    model.eval()
    desc= ' -(Valid)- '
    total_loss, avg_loss, avg_acc = 0, 0, 0.
    iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time= 0, 0, 0, 0
    total_loc, total_cor_loc = 0, 0
    eva_metric = np.zeros((6, 1))

    len_queue = len(valid_queue)
    len_batch = int(np.ceil((len_queue/batch_size)))
    valid_queue = [[valid_queue.popleft() for _ in range(min(batch_size, len(valid_queue)))] for k in range(len_batch)]
    
    with torch.no_grad():
        for i, batch_queue in enumerate(valid_queue):

            max_place = max([len(valid_data[u][idx]['loc']) for u,idx in batch_queue]) 
            max_history = max([len(valid_data[u][idx]['history_loc']) for u,idx in batch_queue]) 

            loc = torch.cat([pad_tensor(valid_data[u][idx]['loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
            time = torch.cat([pad_tensor(valid_data[u][idx]['tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
            loc_label = torch.cat([pad_tensor(valid_data[u][idx]['target_loc'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)
            time_label = torch.cat([pad_tensor(valid_data[u][idx]['target_tim'],max_place,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long().contiguous().view(-1)

            history_loc = torch.cat([before_pad_tensor(valid_data[u][idx]['history_loc'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()
            history_time = torch.cat([before_pad_tensor(valid_data[u][idx]['history_tim'],max_history,0).unsqueeze(0) for u, idx in batch_queue], dim=0).to(device).long()    

            # input_loc = torch.cat([history_loc, loc], dim=1)
            # input_tim = torch.cat([history_time, time], dim=1)
            
            # place_logit = model(input_loc, input_tim, max_history)
            place_logit = model(history_loc, history_time, loc, time)

            loss, n_loc, n_cor_loc, n_cor_time = cal_loss_performance(logit1=place_logit, label1=loc_label, Predict=True)
            eva_metric = get_acc(loc_label, place_logit, eva_metric)

            total_loss += loss.item()
            iter_100_loss += loss.item()
            avg_loss = total_loss/(i+1)

            total_loc += n_loc
            iter_100_loc += n_loc
            total_cor_loc += n_cor_loc
            iter_100_cor_loc += n_cor_loc
            iter_100_cor_time += n_cor_time
            avg_acc = 100.*total_cor_loc/total_loc
            
            if i % 100 == 0:
                print("{} epoch: {:_>2d} | iter: {:_>4d}/{:_>4d} | loss: {:<10.7f} | avg_loss: {:<10.7f} | acc: {:<4.4f} % | avg_acc: {:<4.4f} % | lr: {:<9.7f}".format(
                    desc, epoch, i, len_batch, iter_100_loss/100., avg_loss, 100.*iter_100_cor_loc/iter_100_loc, avg_acc, optimizer._print_lr()))   
                iter_100_loss, iter_100_loc, iter_100_cor_loc, iter_100_cor_time = 0, 0, 0, 0

    return avg_loss, avg_acc, eva_metric/total_loc
