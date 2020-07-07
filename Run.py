import time
import torch
import pickle
import argparse
from Cofig import parsers
from Model.Model import make_model
from Train import train_epoch, valid_epoch, save_model
from Tool_Gen_Data import generate_input_history, generate_queue, pad_tensor, before_pad_tensor
from Optimizer import Trans_Optim
from Eval import cal_loss_performance

def run(epoch, model, optimizer, device, train_data, train_traj_idx, valid_data, valid_traj_idx, log=None, batch_size=4):

    # with SummaryWriter() as writer:

    log_train_file, log_valid_file = None, None

    if log:
        log_train_file = log + '.train.log'
        log_valid_file = log + '.valid.log'
        # print('[Info] Training performance will be written to file: {} and {}'.format(
        #     log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write("\n# Note: .\n")
            log_vf.write("\n# Note: .\n")
            log_tf.write("Start Time: {}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            log_vf.write("Start Time: {}.\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        
    for epoch_i in range(1, epoch+1):
        
        train_queue = generate_queue(train_traj_idx,'random','train')
        valid_queue = generate_queue(valid_traj_idx,'normal','valid') 

        train_avg_loss, train_acc, train_metric = train_epoch(epoch_i, model, train_data, train_queue, optimizer, device, batch_size)
        valid_avg_loss, valid_acc, valid_metric = valid_epoch(epoch_i, model, valid_data, valid_queue, optimizer, device, batch_size)
        print('-'*150)
        print(" --Train--  Epoch: {}/{}  Train_avg_loss: {:<10.7f} Train_acc: {:<4.4f}".format(epoch_i, epoch, train_avg_loss, train_acc))
        print(" --Valid--  Epoch: {}/{}  Valid_avg_loss: {:<10.7f} Valid_acc: {:<4.4f}".format(epoch_i, epoch, valid_avg_loss, valid_acc))
        print('-'*150)
        print(" --Train--  Epoch: {}/{}  Metric: {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f}".format(epoch_i, epoch, train_metric[0][0], train_metric[1][0], train_metric[2][0], train_metric[3][0], train_metric[4][0], train_metric[5][0]))
        print(" --Valid--  Epoch: {}/{}  Metric: {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f} \ {:<4.4f}".format(epoch_i, epoch, valid_metric[0][0], valid_metric[1][0], valid_metric[2][0], valid_metric[3][0], valid_metric[4][0], valid_metric[5][0]))
        print('-'*150)
        
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(" --Train--  Epoch: {}/{}  Train_avg_loss: {} Train_acc: {}\n".format(epoch_i, epoch, train_avg_loss, train_acc))
                log_vf.write(" --Valid--  Epoch: {}/{}  Valid_avg_loss: {} Valid_acc: {}\n".format(epoch_i, epoch, valid_avg_loss, valid_acc)) 
        
        # if epoch_i % 5==0:
        #     save_model(epoch_i, model, Predict=True)
        #     print("The step is {} .".format(optimizer._print_step()))
        #     print('-'*150)
            # writer.add_scalars("Loss", {"Train": train_total_loss, "Valid": valid_total_loss}, epoch_i)
            # writer.add_scalars("Acc", {"Train": train_epoch_acc, "Valid": valid_epoch_acc}, epoch_i)
            # writer.add_scalars("Lr", {"Train": optimizer._print_lr()}, epoch_i)


def main(Bert_Pretrain=False, Pretrained=False, log='predict'):
    print('*'*150)
    print("Get Config")
    args = parsers()
    for key in args.__dict__:
        print(f"{key}:{args.__dict__[key]}")
    print('*'*150)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Get Dataset")
    dataset_4qs = pickle.load(open('./data/tweets-cikm.txtfoursquare.pk', 'rb'))
    print("User Number: ", len(dataset_4qs['uid_list']))
    print("Generate Train_traj_list")
    train_data, train_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="train")
    print("Generate Valid_traj_list")
    test_data, test_traj_idx = generate_input_history(data_neural=dataset_4qs["data_neural"], mode="test")
    print("Get Model")
    model = make_model(token_size=len(dataset_4qs['vid_list']), N=args.head_n, d_model=args.d_model, d_ff=args.d_ff, h=args.head_n, dropout=args.dropout)
    if Pretrained:
        print("Load Pretrained Predict Model")
        model.load_state_dict(torch.load('./pretrain/Predict_model_trained_ep75.pth'))
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer = Trans_Optim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                init_lr=args.lr, d_model=args.d_model, n_warmup_steps=args.n_warmup_steps)
    print('*'*150)
    print('-'*65 + "  START TRAIN  " + '-'*65)
    run(args.epoch, model, optimizer, device, train_data, train_traj_idx, test_data, test_traj_idx, log, args.batch_size)

if __name__ == "__main__":
    main()
    pass