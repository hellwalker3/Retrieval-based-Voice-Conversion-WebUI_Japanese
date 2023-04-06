from multiprocessing import cpu_count
import threading
from time import sleep
from subprocess import Popen
from time import sleep
import torch, os,traceback,sys,warnings,shutil,numpy as np
import faiss
#判断是否有能用来训练和加速推理的N卡
ncpu=cpu_count()
ngpu=torch.cuda.device_count()
gpu_infos=[]
if(torch.cuda.is_available()==False or ngpu==0):if_gpu_ok=False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name=torch.cuda.get_device_name(i)
        if("16"in gpu_name or "MX"in gpu_name):continue
        if("10"in gpu_name or "20"in gpu_name or "30"in gpu_name or "40"in gpu_name or "A50"in gpu_name.upper() or "70"in gpu_name or "80"in gpu_name or "90"in gpu_name or "M4"in gpu_name or "T4"in gpu_name or "TITAN"in gpu_name.upper()):#A10#A100#V100#A40#P40#M40#K80
            if_gpu_ok=True#至少有一张能用的N卡
            gpu_infos.append("%s\t%s"%(i,gpu_name))
gpu_info="\n".join(gpu_infos)if if_gpu_ok==True and len(gpu_infos)>0 else "很遗憾您这没有能用的显卡来支持您训练"
gpus="-".join([i[0]for i in gpu_infos])
now_dir=os.getcwd()
sys.path.append(now_dir)
tmp=os.path.join(now_dir,"TEMP")
shutil.rmtree(tmp,ignore_errors=True)
os.makedirs(tmp,exist_ok=True)
os.makedirs(os.path.join(now_dir,"logs"),exist_ok=True)
os.makedirs(os.path.join(now_dir,"weights"),exist_ok=True)
os.environ["TEMP"]=tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from scipy.io import wavfile
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import is_half,device,is_half,python_cmd,listen_port,iscolab,noparallel
from infer_uvr5 import _audio_pre_
from my_utils import load_audio
from train.process_ckpt import show_info,change_info,merge,extract_small_model
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger('numba').setLevel(logging.WARNING)

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""
    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)
    def get_block_name(self):
        return "button"

hubert_model=None
def load_hubert():
    global hubert_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if(is_half):hubert_model = hubert_model.half()
    else:hubert_model = hubert_model.float()
    hubert_model.eval()

weight_root="weights"
weight_uvr5_root="uvr5_weights"
names=[]
for name in os.listdir(weight_root):
    if name.endswith(".pth"): names.append(name)
uvr5_names=[]
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth"): uvr5_names.append(name.replace(".pth",""))

def vc_single(sid,input_audio,f0_up_key,f0_file,f0_method,file_index,file_big_npy,index_rate):#spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr,net_g,vc,hubert_model
    if input_audio is None:return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio=load_audio(input_audio,16000)
        times = [0, 0, 0]
        if(hubert_model==None):load_hubert()
        if_f0 = cpt.get("f0", 1)
        audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
        print(times)
        return "Success", (tgt_sr, audio_opt)
    except:
        info=traceback.format_exc()
        print(info)
        return info,(None,None)

def vc_multi(sid,dir_path,opt_root,paths,f0_up_key,f0_method,file_index,file_big_npy,index_rate):
    try:
        dir_path=dir_path.strip(" ")#防止小白拷路径头尾带了空格
        opt_root=opt_root.strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if(dir_path!=""):paths=[os.path.join(dir_path,name)for name in os.listdir(dir_path)]
            else:paths=[path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos=[]
        for path in paths:
            info,opt=vc_single(sid,path,f0_up_key,None,f0_method,file_index,file_big_npy,index_rate)
            if(info=="Success"):
                try:
                    tgt_sr,audio_opt=opt
                    wavfile.write("%s/%s" % (opt_root, os.path.basename(path)), tgt_sr, audio_opt)
                except:
                    info=traceback.format_exc()
            infos.append("%s->%s"%(os.path.basename(path),info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()

def uvr(model_name,inp_root,save_root_vocal,paths,save_root_ins):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip("\n")
        save_root_vocal = save_root_vocal.strip(" ").strip("\n")
        save_root_ins = save_root_ins.strip(" ").strip("\n")
        pre_fun = _audio_pre_(model_path=os.path.join(weight_uvr5_root,model_name+".pth"), device=device, is_half=is_half)
        if (inp_root != ""):paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:paths = [path.name for path in paths]
        for name in paths:
            inp_path=os.path.join(inp_root,name)
            try:
                pre_fun._path_audio_(inp_path , save_root_ins,save_root_vocal)
                infos.append("%s->Success"%(os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append("%s->%s" % (os.path.basename(inp_path),traceback.format_exc()))
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            del pre_fun.model
            del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        torch.cuda.empty_cache()
    yield "\n".join(infos)

#一个选项卡全局只能有一个音色
def get_vc(sid):
    global n_spk,tgt_sr,net_g,vc,cpt
    if(sid==""):
        global hubert_model
        print("clean_empty_cache")
        del net_g, n_spk, vc, hubert_model,tgt_sr#,cpt
        hubert_model = net_g=n_spk=vc=hubert_model=tgt_sr=None
        torch.cuda.empty_cache()
        ###楼下不这么折腾清理不干净
        if_f0 = cpt.get("f0", 1)
        if (if_f0 == 1):
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del net_g,cpt
        torch.cuda.empty_cache()
        cpt=None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s"%person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3]=cpt["weight"]["emb_g.weight"].shape[0]#n_spk
    if_f0=cpt.get("f0",1)
    if(if_f0==1):
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
    else:
        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(device)
    if (is_half):net_g = net_g.half()
    else:net_g = net_g.float()
    vc = VC(tgt_sr, device, is_half)
    n_spk=cpt["config"][-3]
    return {"visible": True,"maximum": n_spk, "__type__": "update"}

def change_choices():
    names=[]
    for name in os.listdir(weight_root):
        if name.endswith(".pth"): names.append(name)
    return {"choices": sorted(names), "__type__": "update"}
def clean():return {"value": "", "__type__": "update"}
def change_f0(if_f0_3,sr2):#np7, f0method8,pretrained_G14,pretrained_D15
    if(if_f0_3=="是"):return {"visible": True, "__type__": "update"},{"visible": True, "__type__": "update"},"pretrained/f0G%s.pth"%sr2,"pretrained/f0D%s.pth"%sr2
    return {"visible": False, "__type__": "update"}, {"visible": False, "__type__": "update"},"pretrained/G%s.pth"%sr2,"pretrained/D%s.pth"%sr2

sr_dict={
    "32k":32000,
    "40k":40000,
    "48k":48000,
}

def if_done(done,p):
    while 1:
        if(p.poll()==None):sleep(0.5)
        else:break
    done[0]=True


def if_done_multi(done,ps):
    while 1:
        #poll==None代表进程未结束
        #只要有一个进程未结束都不停
        flag=1
        for p in ps:
            if(p.poll()==None):
                flag = 0
                sleep(0.5)
                break
        if(flag==1):break
    done[0]=True

def preprocess_dataset(trainset_dir,exp_dir,sr,n_p=ncpu):
    sr=sr_dict[sr]
    os.makedirs("%s/logs/%s"%(now_dir,exp_dir),exist_ok=True)
    f = open("%s/logs/%s/preprocess.log"%(now_dir,exp_dir), "w")
    f.close()
    cmd=python_cmd + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "%(trainset_dir,sr,n_p,now_dir,exp_dir)+str(noparallel)
    print(cmd)
    p = Popen(cmd, shell=True)#, stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr，popen read都非得全跑完了再一次性读取，不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done=[False]
    threading.Thread(target=if_done,args=(done,p,)).start()
    while(1):
        with open("%s/logs/%s/preprocess.log"%(now_dir,exp_dir),"r")as f:yield(f.read())
        sleep(1)
        if(done[0]==True):break
    with open("%s/logs/%s/preprocess.log"%(now_dir,exp_dir), "r")as f:log = f.read()
    print(log)
    yield log
#but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus,n_p,f0method,if_f0,exp_dir):
    gpus=gpus.split("-")
    os.makedirs("%s/logs/%s"%(now_dir,exp_dir),exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir), "w")
    f.close()
    if(if_f0=="是"):
        cmd=python_cmd + " extract_f0_print.py %s/logs/%s %s %s"%(now_dir,exp_dir,n_p,f0method)
        print(cmd)
        p = Popen(cmd, shell=True,cwd=now_dir)#, stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr，popen read都非得全跑完了再一次性读取，不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done=[False]
        threading.Thread(target=if_done,args=(done,p,)).start()
        while(1):
            with open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir),"r")as f:yield(f.read())
            sleep(1)
            if(done[0]==True):break
        with open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir), "r")as f:log = f.read()
        print(log)
        yield log
    ####对不同part分别开多进程
    '''
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    '''
    leng=len(gpus)
    ps=[]
    for idx,n_g in enumerate(gpus):
        cmd=python_cmd + " extract_feature_print.py %s %s %s %s/logs/%s"%(leng,idx,n_g,now_dir,exp_dir)
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)#, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr，popen read都非得全跑完了再一次性读取，不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps,)).start()
    while (1):
        with open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir), "r")as f:yield (f.read())
        sleep(1)
        if (done[0] == True): break
    with open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir), "r")as f:log = f.read()
    print(log)
    yield log
def change_sr2(sr2,if_f0_3):
    if(if_f0_3=="是"):return "pretrained/f0G%s.pth"%sr2,"pretrained/f0D%s.pth"%sr2
    else:return "pretrained/G%s.pth"%sr2,"pretrained/D%s.pth"%sr2
#but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17):
    #生成filelist
    exp_dir="%s/logs/%s"%(now_dir,exp_dir1)
    os.makedirs(exp_dir,exist_ok=True)
    gt_wavs_dir="%s/0_gt_wavs"%(exp_dir)
    co256_dir="%s/3_feature256"%(exp_dir)
    if(if_f0_3=="是"):
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir="%s/2b-f0nsf"%(exp_dir)
        names=set([name.split(".")[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(".")[0]for name in os.listdir(co256_dir)])&set([name.split(".")[0]for name in os.listdir(f0_dir)])&set([name.split(".")[0]for name in os.listdir(f0nsf_dir)])
    else:
        names=set([name.split(".")[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(".")[0]for name in os.listdir(co256_dir)])
    opt=[]
    for name in names:
        if (if_f0_3 == "是"):
            opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"%(gt_wavs_dir.replace("\\","\\\\"),name,co256_dir.replace("\\","\\\\"),name,f0_dir.replace("\\","\\\\"),name,f0nsf_dir.replace("\\","\\\\"),name,spk_id5))
        else:
            opt.append("%s/%s.wav|%s/%s.npy|%s"%(gt_wavs_dir.replace("\\","\\\\"),name,co256_dir.replace("\\","\\\\"),name,spk_id5))
    with open("%s/filelist.txt"%exp_dir,"w")as f:f.write("\n".join(opt))
    print("write filelist done")
    #生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:",gpus16)
    if gpus16:
        cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s" % (exp_dir1,sr2,1 if if_f0_3=="是"else 0,batch_size12,gpus16,total_epoch11,save_epoch10,pretrained_G14,pretrained_D15,1 if if_save_latest13=="是"else 0,1 if if_cache_gpu17=="是"else 0)
    else:
        cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s" % (exp_dir1,sr2,1 if if_f0_3=="是"else 0,batch_size12,total_epoch11,save_epoch10,pretrained_G14,pretrained_D15,1 if if_save_latest13=="是"else 0,1 if if_cache_gpu17=="是"else 0)
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束，您可查看控制台训练日志或实验文件夹下的train.log"
# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1):
    exp_dir="%s/logs/%s"%(now_dir,exp_dir1)
    os.makedirs(exp_dir,exist_ok=True)
    feature_dir="%s/3_feature256"%(exp_dir)
    if(os.path.exists(feature_dir)==False):return "请先进行特征提取!"
    listdir_res=list(os.listdir(feature_dir))
    if(len(listdir_res)==0):return "请先进行特征提取！"
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    np.save("%s/total_fea.npy"%exp_dir, big_npy)
    n_ivf = big_npy.shape[0] // 39
    infos=[]
    infos.append("%s,%s"%(big_npy.shape,n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256, "IVF%s,Flat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = int(np.power(n_ivf,0.3))
    index.train(big_npy)
    faiss.write_index(index, '%s/trained_IVF%s_Flat_nprobe_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe))
    infos.append("adding")
    yield "\n".join(infos)
    index.add(big_npy)
    faiss.write_index(index, '%s/added_IVF%s_Flat_nprobe_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe))
    infos.append("成功构建索引，added_IVF%s_Flat_nprobe_%s.index"%(n_ivf,index_ivf.nprobe))
    yield "\n".join(infos)
#but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17):
    infos=[]
    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)
    os.makedirs("%s/logs/%s"%(now_dir,exp_dir1),exist_ok=True)
    #########step1:处理数据
    open("%s/logs/%s/preprocess.log"%(now_dir,exp_dir1), "w").close()
    cmd=python_cmd + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "%(trainset_dir4,sr_dict[sr2],ncpu,now_dir,exp_dir1)+str(noparallel)
    yield get_info_str("step1:正在处理数据")
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir1), "r")as f: print(f.read())
    #########step2a:提取音高
    open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir1), "w")
    if(if_f0_3=="是"):
        yield get_info_str("step2a:正在提取音高")
        cmd=python_cmd + " extract_f0_print.py %s/logs/%s %s %s"%(now_dir,exp_dir1,np7,f0method8)
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True,cwd=now_dir)
        p.wait()
        with open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir1), "r")as f:print(f.read())
    else:yield get_info_str("step2a:无需提取音高")
    #######step2b:提取特征
    yield get_info_str("step2b:正在提取特征")
    gpus=gpus16.split("-")
    leng=len(gpus)
    ps=[]
    for idx,n_g in enumerate(gpus):
        cmd=python_cmd + " extract_feature_print.py %s %s %s %s/logs/%s"%(leng,idx,n_g,now_dir,exp_dir1)
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)#, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:p.wait()
    with open("%s/logs/%s/extract_f0_feature.log"%(now_dir,exp_dir1), "r")as f:print(f.read())
    #######step3a:训练模型
    yield get_info_str("step3a:正在训练模型")
    #生成filelist
    exp_dir="%s/logs/%s"%(now_dir,exp_dir1)
    gt_wavs_dir="%s/0_gt_wavs"%(exp_dir)
    co256_dir="%s/3_feature256"%(exp_dir)
    if(if_f0_3=="是"):
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir="%s/2b-f0nsf"%(exp_dir)
        names=set([name.split(".")[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(".")[0]for name in os.listdir(co256_dir)])&set([name.split(".")[0]for name in os.listdir(f0_dir)])&set([name.split(".")[0]for name in os.listdir(f0nsf_dir)])
    else:
        names=set([name.split(".")[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(".")[0]for name in os.listdir(co256_dir)])
    opt=[]
    for name in names:
        if (if_f0_3 == "是"):
            opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"%(gt_wavs_dir.replace("\\","\\\\"),name,co256_dir.replace("\\","\\\\"),name,f0_dir.replace("\\","\\\\"),name,f0nsf_dir.replace("\\","\\\\"),name,spk_id5))
        else:
            opt.append("%s/%s.wav|%s/%s.npy|%s"%(gt_wavs_dir.replace("\\","\\\\"),name,co256_dir.replace("\\","\\\\"),name,spk_id5))
    with open("%s/filelist.txt"%exp_dir,"w")as f:f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s" % (exp_dir1,sr2,1 if if_f0_3=="是"else 0,batch_size12,gpus16,total_epoch11,save_epoch10,pretrained_G14,pretrained_D15,1 if if_save_latest13=="是"else 0,1 if if_cache_gpu17=="是"else 0)
    else:
        cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s" % (exp_dir1,sr2,1 if if_f0_3=="是"else 0,batch_size12,total_epoch11,save_epoch10,pretrained_G14,pretrained_D15,1 if if_save_latest13=="是"else 0,1 if if_cache_gpu17=="是"else 0)
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str("训练结束，您可查看控制台训练日志或实验文件夹下的train.log")
    #######step3b:训练索引
    feature_dir="%s/3_feature256"%(exp_dir)
    npys = []
    listdir_res=list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    np.save("%s/total_fea.npy"%exp_dir, big_npy)
    n_ivf = big_npy.shape[0] // 39
    yield get_info_str("%s,%s"%(big_npy.shape,n_ivf))
    index = faiss.index_factory(256, "IVF%s,Flat"%n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = int(np.power(n_ivf,0.3))
    index.train(big_npy)
    faiss.write_index(index, '%s/trained_IVF%s_Flat_nprobe_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe))
    yield get_info_str("adding index")
    index.add(big_npy)
    faiss.write_index(index, '%s/added_IVF%s_Flat_nprobe_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe))
    yield get_info_str("成功构建索引，added_IVF%s_Flat_nprobe_%s.index"%(n_ivf,index_ivf.nprobe))
    yield get_info_str("全流程结束！")

#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if(os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path),"train.log"))==False):return {"__type__": "update"},{"__type__": "update"}
    try:
        with open(ckpt_path.replace(os.path.basename(ckpt_path),"train.log"),"r")as f:
            info=eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr,f0=info["sample_rate"],info["if_f0"]
            return sr,str(f0)
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}


with gr.Blocks() as app:
    gr.Markdown(value="""
        このソフトウェアはMITライセンスでオープンソース化されており、作者はソフトウェアに対して何らかの制御力を持っていません。ソフトウェアの使用者や、ソフトウェアによって生成された音声を拡散する者は、すべての責任を自分で負うものとします。<br>
        この条件に同意できない場合は、ソフトウェアパッケージ内のすべてのコードやファイルを使用または参照することはできません。詳細はルートディレクトリの"使用に従ってください-LICENSE.txt"を参照してください。
        """)
    with gr.Tabs():
        with gr.TabItem("モデル推論"):
            with gr.Row():
                sid0 = gr.Dropdown(label="推論音色", choices=sorted(names))
                refresh_button = gr.Button("音色リストを更新", variant="primary")
                refresh_button.click(
                    fn=change_choices,
                    inputs=[],
                    outputs=[sid0]
                )
                clean_button = gr.Button("音色をアンロードしてメモリを節約", variant="primary")
                spk_item = gr.Slider(minimum=0, maximum=2333, step=1, label='話者IDを選択してください', value=0, visible=False, interactive=True)
                clean_button.click(
                    fn=clean,
                    inputs=[],
                    outputs=[sid0]
                )
                sid0.change(
                    fn=get_vc,
                    inputs=[sid0],
                    outputs=[spk_item],
                )
            with gr.Group():
                gr.Markdown(value="""
                    男性から女性への推奨+12キー、女性から男性への推奨-12キー、音域が破裂して音色が歪む場合は、適切な音域に自分で調整してください。
                    """)
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(label="ピッチ変更（整数、半音数、オクターブアップ12、オクターブダウン-12）", value=0)
                        input_audio0 = gr.Textbox(label="処理する音声ファイルのパスを入力してください（デフォルトは正しい形式の例）",value="E:\codes\py39\\vits_vc_gpu_train\\todo-songs\冬之花clip1.wav")
                        f0method0=gr.Radio(label="音高抽出アルゴリズムを選択してください。歌声入力にはpmで高速化可能。harvestは低音が良いが非常に遅い。", choices=["pm","harvest"],value="pm", interactive=True)
                    with gr.Column():
                        file_index1 = gr.Textbox(label="特徴検索ライブラリファイルパス",value="E:\codes\py39\\vits_vc_gpu_train\logs\mi-test-1key\\added_IVF677_Flat_nprobe_7.index", interactive=True)
                        file_big_npy1 = gr.Textbox(label="特徴ファイルパス",value="E:\codes\py39\\vits_vc_gpu_train\logs\mi-test-1key\\total_fea.npy", interactive=True)
                        index_rate1 =  gr.Slider(minimum=0, maximum=1,label='検索特徴の割合', value=1,interactive=True)
                    f0_file = gr.File(label="F0曲線ファイル、オプション、1行1音高、デフォルトのF0およびピッチ変更を置き換え")
                    but0=gr.Button("変換", variant="primary")
                    with gr.Column():
                        vc_output1 = gr.Textbox(label="出力情報")
                        vc_output2 = gr.Audio(label="出力音声（右下隅の3つの点をクリックしてダウンロード）")
                    but0.click(vc_single, [spk_item, input_audio0, vc_transform0,f0_file,f0method0,file_index1,file_big_npy1,index_rate1], [vc_output1, vc_output2])
            with gr.Group():
                gr.Markdown(value="""
                    一括変換。変換する音声ファイルのフォルダを入力するか、複数の音声ファイルをアップロードしてください。指定されたフォルダ（デフォルトはopt）に変換された音声を出力します。
                    """)
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(label="ピッチ変更（整数、半音数、オクターブアップ12、オクターブダウン-12）", value=0)
                        opt_input = gr.Textbox(label="出力フォルダを指定",value="opt")
                        f0method1=gr.Radio(label="音高抽出アルゴリズムを選択してください。歌声入力にはpmで高速化可能。harvestは低音が良いが非常に遅い。", choices=["pm","harvest"],value="pm", interactive=True)

                    with gr.Column():
                        file_index2 = gr.Textbox(label="特徴検索ライブラリファイルパス",value="E:\codes\py39\\vits_vc_gpu_train\logs\mi-test-1key\\added_IVF677_Flat_nprobe_7.index", interactive=True)
                        file_big_npy2 = gr.Textbox(label="特徴ファイルパス",value="E:\codes\py39\\vits_vc_gpu_train\logs\mi-test-1key\\total_fea.npy", interactive=True)
                        index_rate2 =  gr.Slider(minimum=0, maximum=1,label='検索特徴の割合', value=1,interactive=True)
                    with gr.Column():
                        dir_input = gr.Textbox(label="処理待ちの音声フォルダのパスを入力（ファイルマネージャのアドレスバーからコピー）",value="E:\codes\py39\\vits_vc_gpu_train\\todo-songs")
                        inputs = gr.File(file_count="multiple", label="または複数の音声ファイルを一括入力できます。二つのうち一つを選択してください。フォルダが優先されます。")
                    but1=gr.Button("変換", variant="primary")
                    vc_output3 = gr.Textbox(label="出力情報")
                    but1.click(vc_multi, [spk_item, dir_input,opt_input,inputs, vc_transform1,f0method1,file_index2,file_big_npy2,index_rate2], [vc_output3])
        with gr.TabItem("伴奏とボーカルの分離"):
            with gr.Group():
                gr.Markdown(value="""
                    UVR5モデルを使用したボーカルと伴奏の分離の一括処理。<br>
                    ハーモニーなしの場合はHP2を使用し、ハーモニーありで抽出されるボーカルにハーモニーが不要な場合はHP5を使用。<br>
                    正しいフォルダパスの例：E:\codes\py39\\vits_vc_gpu\白鹭霜华テストサンプル（ファイルマネージャのアドレスバーからコピー）
                    """)
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(label="処理待ちの音声フォルダのパスを入力",value="E:\codes\py39\\vits_vc_gpu_train\\todo-songs")
                        wav_inputs = gr.File(file_count="multiple", label="または複数の音声ファイルを一括入力できます。二つのうち一つを選択してください。フォルダが優先されます。")
                    with gr.Column():
                        model_choose = gr.Dropdown(label="モデル", choices=uvr5_names)
                        opt_vocal_root = gr.Textbox(label="ボーカルの出力フォルダを指定",value="opt")
                        opt_ins_root = gr.Textbox(label="楽器の出力フォルダを指定",value="opt")
                    but2=gr.Button("変換", variant="primary")
                    vc_output4 = gr.Textbox(label="出力情報")
                    but2.click(uvr, [model_choose, dir_wav_input,opt_vocal_root,wav_inputs,opt_ins_root], [vc_output4])
        with gr.TabItem("トレーニング"):
            gr.Markdown(value="""
                step1：実験設定を記入します。実験データはlogsに格納され、各実験ごとに1つのフォルダがあり、手動で実験名のパスを入力し、実験設定、ログ、トレーニングで得られたモデルファイルが含まれます。
                """)
            with gr.Row():
                exp_dir1 = gr.Textbox(label="実験名を入力",value="mi-test")
                sr2 = gr.Radio(label="目標サンプリングレート", choices=["32k","40k","48k"],value="40k", interactive=True)
                if_f0_3 = gr.Radio(label="モデルに音高ガイドが必要か（歌唱は必須、音声は不要）", choices=["是","否"],value="是", interactive=True)
            with gr.Group():# 一時的に1人分のみ対応、後で最大4人まで対応#データ処理
                gr.Markdown(value="""
                    step2a：トレーニングフォルダ内のすべてのデコード可能なオーディオファイルを自動的に走査し、スライスと正規化を行い、実験ディレクトリに2つのwavフォルダを生成します。現時点では、1人分のトレーニングのみをサポートしています。
                    """)
                with gr.Row():
                    trainset_dir4 = gr.Textbox(label="トレーニングフォルダのパスを入力",value="E:\音声音源+アノテーション\米津玄師\src")
                    spk_id5 = gr.Slider(minimum=0, maximum=4, step=1, label='話者IDを指定してください', value=0,interactive=True)
                    but1=gr.Button("データ処理", variant="primary")
                    info1=gr.Textbox(label="出力情報",value="")
                    but1.click(preprocess_dataset,[trainset_dir4,exp_dir1,sr2],[info1])
            with gr.Group():
                gr.Markdown(value="""
                    step2b：CPUを使用して音高を抽出（音高ガイド付きモデルの場合）、GPUを使用して特徴抽出（カード番号を選択）
                    """)
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(label="使用するカード番号を-で区切って入力。例： 0-1-2   カード0とカード1とカード2を使用",value=gpus,interactive=True)
                        gpu_info9 = gr.Textbox(label="GPU情報",value=gpu_info)
                    with gr.Column():
                        np7 = gr.Slider(minimum=0, maximum=ncpu, step=1, label='音高抽出に使用するCPUプロセス数', value=ncpu,interactive=True)
                        f0method8 = gr.Radio(label="音高抽出アルゴリズムの選択：歌声の入力でpmで高速化可能、高品質な音声だがCPUが劣る場合dioで高速化可能、harvestは品質が良いが遅い", choices=["pm", "harvest","dio"], value="harvest", interactive=True)
                    but2=gr.Button("特徴抽出", variant="primary")
                    info2=gr.Textbox(label="出力情報",value="",max_lines=8)
                    but2.click(extract_f0_feature,[gpus6,np7,f0method8,if_f0_3,exp_dir1],[info2])
            with gr.Group():
                gr.Markdown(value="""
                    step3：トレーニング設定を入力し、モデルとインデックスのトレーニングを開始
                    """)
                with gr.Row():
                    save_epoch10 = gr.Slider(minimum=0, maximum=50, step=1, label='保存頻度save_every_epoch', value=5,interactive=True)
                    total_epoch11 = gr.Slider(minimum=0, maximum=100, step=1, label='トータルエポック数total_epoch', value=10,interactive=True)
                    batch_size12 = gr.Slider(minimum=0, maximum=32, step=1, label='batch_size', value=4,interactive=True)
                    if_save_latest13 = gr.Radio(label="最新のckptファイルのみを保存してディスク容量を節約するか", choices=["是","否"],value="否", interactive=True)
                    if_cache_gpu17 = gr.Radio(label="すべてのトレーニングセットをグラフィックメモリにキャッシュするか。10分以下の小規模データはキャッシュしてトレーニングを加速できますが、大規模データはキャッシュがグラフィックメモリを圧迫し、速度向上はあまり期待できません", choices=["是","否"],value="否", interactive=True)
                with gr.Row():
                    pretrained_G14 = gr.Textbox(label="事前学習済みモデルGのパスを入力", value="pretrained/f0G40k.pth",interactive=True)
                    pretrained_D15 = gr.Textbox(label="事前学習済みモデルDのパスを入力", value="pretrained/f0D40k.pth",interactive=True)
                    sr2.change(change_sr2, [sr2,if_f0_3], [pretrained_G14,pretrained_D15])
                    if_f0_3.change(change_f0, [if_f0_3, sr2], [np7, f0method8, pretrained_G14, pretrained_D15])
                    gpus16 = gr.Textbox(label="使用するカード番号を-で区切って入力。例： 0-1-2   カード0とカード1とカード2を使用", value=gpus,interactive=True)
                    but3 = gr.Button("モデルの訓練", variant="primary")
                    but4 = gr.Button("特徴インデックスの訓練", variant="primary")
                    but5 = gr.Button("ワンキー訓練", variant="primary")
                    info3 = gr.Textbox(label="出力情報", value="",max_lines=10)
                    but3.click(click_train,[exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17],info3)
                    but4.click(train_index,[exp_dir1],info3)
                    but5.click(train1key,[exp_dir1,sr2,if_f0_3,trainset_dir4,spk_id5,gpus6,np7,f0method8,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17],info3)

        with gr.TabItem("ckpt処理"):
            with gr.Group():
                gr.Markdown(value="""モデルの統合。音色の統合テストに使用できます""")
                with gr.Row():
                    ckpt_a = gr.Textbox(label="モデルAのパス", value="", interactive=True)
                    ckpt_b = gr.Textbox(label="モデルBのパス", value="", interactive=True)
                    alpha_a = gr.Slider(minimum=0, maximum=1, label='モデルAの重み', value=0.5, interactive=True)
                with gr.Row():
                    sr_ = gr.Radio(label="目標サンプリングレート", choices=["32k","40k","48k"],value="40k", interactive=True)
                    if_f0_ = gr.Radio(label="モデルに音高ガイダンスが含まれているか", choices=["是","否"],value="是", interactive=True)
                    info__ = gr.Textbox(label="埋め込むモデル情報", value="", max_lines=8, interactive=True)
                    name_to_save0=gr.Textbox(label="拡張子なしの保存モデル名", value="", max_lines=1, interactive=True)
                with gr.Row():
                    but6 = gr.Button("統合", variant="primary")
                    info4 = gr.Textbox(label="出力情報", value="", max_lines=8)
                but6.click(merge, [ckpt_a,ckpt_b,alpha_a,sr_,if_f0_,info__,name_to_save0], info4)
            with gr.Group():
                gr.Markdown(value="モデル情報の変更（weightsフォルダから抽出された小さなモデルファイルのみ対応）")
                with gr.Row():
                    ckpt_path0 = gr.Textbox(label="モデルのパス", value="", interactive=True)
                    info_=gr.Textbox(label="変更するモデル情報", value="", max_lines=8, interactive=True)
                    name_to_save1=gr.Textbox(label="デフォルトで空白の保存ファイル名は、元のファイルと同じ名前になります", value="", max_lines=8, interactive=True)
                with gr.Row():
                    but7 = gr.Button("変更", variant="primary")
                    info5 = gr.Textbox(label="出力情報", value="", max_lines=8)
                but7.click(change_info, [ckpt_path0,info_,name_to_save1], info5)
            with gr.Group():
                gr.Markdown(value="モデル情報の表示（weightsフォルダから抽出された小さなモデルファイルのみ対応）")
                with gr.Row():
                    ckpt_path1 = gr.Textbox(label="モデルのパス", value="", interactive=True)
                    but8 = gr.Button("表示", variant="primary")
                    info6 = gr.Textbox(label="出力情報", value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6)
            with gr.Group():
                gr.Markdown(value="モデル抽出（logsフォルダ内の大きなファイルモデルのパスを入力）、途中で訓練をやめたくない場合や、中間モデルをテストしたい場合に適しています")
                with gr.Row():
                    ckpt_path2 = gr.Textbox(label="モデルのパス", value="E:\codes\py39\logs\mi-test_f0_48k\\G_23333.pth", interactive=True)
                    save_name = gr.Textbox(label="保存名", value="", interactive=True)
                    sr__ = gr.Radio(label="目標サンプリングレート", choices=["32k","40k","48k"],value="40k", interactive=True)
                    if_f0__ = gr.Radio(label="モデルに音高ガイダンスが含まれているか, 1はい0いいえ", choices=["1","0"],value="1", interactive=True)
                    info___ = gr.Textbox(label="埋め込むモデル情報", value="", max_lines=8, interactive=True)
                    but9 = gr.Button("抽出", variant="primary")
                    info7 = gr.Textbox(label="出力情報", value="", max_lines=8)
                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
                but9.click(extract_small_model, [ckpt_path2,save_name,sr__,if_f0__,info___], info7)

        with gr.TabItem("音高曲線フロントエンドエディター募集"):
            gr.Markdown(value="""開発グループに連絡してくださいxxxxx""")
        with gr.TabItem("クリックしてコミュニケーション・問題フィードバックグループ番号を表示"):
            gr.Markdown(value="""xxxxx""")


    if iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(server_name="0.0.0.0",inbrowser=True,server_port=listen_port,quiet=True)
