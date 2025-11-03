import torch
from diffusers import FluxPipeline
from src.configs.config import get_cfg
from src.data import loader as data_loader
import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader

from launch import default_argument_parser, logging_train_setup
import os
from src.utils.file_io import PathManager
import numpy as np
import random
from src.solver.losses import build_loss
from src.solver.lr_scheduler import make_scheduler
from src.solver.optimizer import make_optimizer
from src.utils.train_utils import AverageMeter
import time
import datetime


from src.models.class_head import classifier

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    #cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    prompt = cfg.MODEL.PROMPT.NUM_TOKENS 
    prompt_type = cfg.MODEL.PROMPT.TYPE
    mlp = cfg.MODEL.MLP_NUM
    t_list = "_" + "_".join(map(str, cfg.MODEL.T_LIST))
    layer_list = "_" + "_".join(map(str, cfg.MODEL.FEATURES_LAYER_LIST))
    feat_list = "_" + "_".join(map(str, cfg.MODEL.FEATURES_TYPE_LIST))
    if cfg.MODEL.FUSION_TYPE == "attention":
        fusion = int(cfg.MODEL.FUSION_ARC.split(',')[0].strip().split(':')[2])
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"t{t_list}_l{layer_list}_type{feat_list}_fusion_{fusion}_lr{lr}_wd{wd}_prompt{prompt_type}{prompt}_mlp{mlp}")
    elif cfg.MODEL.FUSION_TYPE == "linear":
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"t{t_list}_l{layer_list}_type{feat_list}_linear__lr{lr}_wd{wd}_prompt{prompt_type}{prompt}_mlp{mlp}")
    elif cfg.MODEL.FUSION_TYPE == "conv":
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"t{t_list}_l{layer_list}_type{feat_list}_conv__lr{lr}_wd{wd}_prompt{prompt_type}{prompt}_mlp{mlp}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        #sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    # if count > cfg.RUN_N_TIMES:
    #     raise ValueError(
    #         f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg

def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        #train_loader = data_loader.construct_trainval_loader(cfg)
        train_loader = data_loader.construct_train_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader

def get_input( data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels



def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    #train_loader, val_loader, test_loader = get_cifar100_dataloader(cfg, logger)
    #test_loader = None
    #train_loader = test_loader
    logger.info("Constructing models...")
    base_model_id = "Freepik/flux.1-lite-8B-alpha"
    torch_dtype = torch.bfloat16
    device = "cuda"

    # Load the pipe
    model_id = "black-forest-labs/FLUX.1-dev" #"black-forest-labs/FLUX.1-dev"#"Freepik/flux.1-lite-8B-alpha"
    pipe = FluxPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype
    ).to(device)

    classifier_head = classifier(cfg).to(device)
    optimizer = make_optimizer([classifier_head], cfg.SOLVER)
    scheduler = make_scheduler(optimizer, cfg.SOLVER)
    cls_criterion = build_loss(cfg)

    #将cfg 传进pipe.transformer
    pipe.transformer.cfg = cfg
    pipe.cfg = cfg

    guidance_scale = 3.5  # Keep guidance_scale at 3.5
    n_steps = 1000

    total_epoch = cfg.SOLVER.TOTAL_EPOCH
    total_data = len(train_loader)
    best_epoch = -1
    best_metric = 0
    acc_test = 0
    log_interval = cfg.SOLVER.LOG_EVERY_N


    losses = AverageMeter('Loss', ':.4e')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    

    for epoch in range(total_epoch):
            losses.reset()
            batch_time.reset()
            data_time.reset()
            lr = scheduler.get_lr()[0]
            logger.info(f"Starting epoch {epoch}/{total_epoch}. with lr {lr:.6f}")
            classifier_head.train()

            end = time.time()
            for idx, input_data in enumerate(train_loader):
                #if idx>10:
                #    break
                X, targets = get_input(input_data)
                X = X.to(device)
                X = X.to(torch_dtype)
                targets = targets.to(device)
                with torch.inference_mode():
                    outputs = pipe(
                    image=X,
                    prompt="",
                    generator=torch.Generator(device="cpu").manual_seed(cfg.SEED),
                    num_inference_steps=n_steps,
                    guidance_scale=guidance_scale,
                    height=cfg.DATA.CROPSIZE,
                    width=cfg.DATA.CROPSIZE,
                )
                #save features
                #for key, features in outputs.items():
                #            if key == "query" or key == "value":
                #                features = features[0].to(torch.float32).cpu().detach().numpy()
                #                np.save(f"/home/cldb5/flux/{cfg.DATA.NAME}/"+f"t_{cfg.MODEL.T_LIST[0]}_b{cfg.MODEL.FEATURES_LAYER_LIST[0]}_img_{0}_{key}.npy", features)
                #break
                #features_list.append(outputs)
                result = classifier_head([outputs])
                loss = cls_criterion(result, targets, None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.update(loss.cpu().item(), X.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            loss.cpu().item()
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                    )
            scheduler.step()
            #eval                
            classifier_head.eval()

            # eval at each epoch for single gpu training
            loss_eval, acc_eval = evaluate(cfg, pipe, classifier_head, val_loader, cls_criterion, logger, device, torch_dtype)
            # if test_loader is not None:
            #     self.eval_classifier(
            #         test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            #t_name = "val_" + val_loader.dataset.name
            t_name = "val_" + cfg.DATA.NAME
            test_name = "test_" + cfg.DATA.NAME

            if acc_eval > best_metric:
                best_metric = acc_eval
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
                if test_loader is not None and acc_eval > cfg.SOLVER.ACC_THRESHOLD:
                    #make sure acc is good enough to run test
                    loss_test, acc_test = evaluate(cfg, pipe, classifier_head, test_loader, cls_criterion, logger, device, torch_dtype)
        
    #save the best results info
    logger.info(f'Best epoch {best_epoch}: val top1: {best_metric:.3f}')
    logger.info(f'test top1: {acc_test:.3f} at epoch {best_epoch}')
    logger.info(f"average train loss: {losses.avg:.4f}")

        
@torch.no_grad()
def evaluate(cfg, pipe, classifier_head, data_loader, criterion, logger, device="cuda", torch_dtype=torch.bfloat16):
    """
    Evaluate classifier performance on a given dataloader (Top-1 accuracy only).

    Args:
        cfg: Config object.
        pipe: FluxPipeline model used for feature extraction.
        classifier_head: Trained classifier head.
        data_loader: DataLoader for validation or test split.
        criterion: Loss function.
        logger: Logger instance.
        device: Device string (e.g., "cuda" or "cpu").
        torch_dtype: Precision type.

    Returns:
        avg_loss (float): Average loss across dataset.
        top1_acc (float): Top-1 accuracy (%).
    """
    classifier_head.eval()
    #pipe.transformer.eval()

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    start = time.time()

    for idx, input_data in enumerate(data_loader):
        #if idx>10:
        #    break
        X, targets = get_input(input_data)
        X = X.to(device)
        X = X.to(torch_dtype)
        targets = targets.to(device)


        # Extract features with the diffusion model

        with torch.inference_mode():
            outputs = pipe(
            image=X,
            prompt="",
            generator=torch.Generator(device="cpu").manual_seed(cfg.SEED),
            num_inference_steps=getattr(cfg, "EVAL_STEPS", 1000),
            guidance_scale=getattr(cfg, "EVAL_GUIDANCE", 3.5),
            height=cfg.DATA.CROPSIZE,
            width=cfg.DATA.CROPSIZE,
        )

        # Forward through classifier
        logits = classifier_head([outputs])
        loss = criterion(logits, targets, None)
        losses.update(loss.cpu().item(), X.size(0))

        # Compute top-1 accuracy
        pred = logits.argmax(dim=1)
        correct = pred.eq(targets).sum().item()
        acc1 = 100.0 * correct / targets.size(0)
        top1.update(acc1, targets.size(0))

    elapsed = time.time() - start
    logger.info(
        f"Finished evaluation: Loss {losses.avg:.4f}, "
        f"Top1 {top1.avg:.2f}%, Time {elapsed:.1f}s"
    )

    return losses.avg, top1.avg


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)