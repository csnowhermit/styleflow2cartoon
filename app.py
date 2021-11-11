import streamlit as st
st.set_page_config(
    layout="wide",  # Can be "centered" or "wide"
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="StyleFlow web demo",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

import os
import pickle
import copy
import numpy as np
import torch
import torch.nn

import dnnlib
import legacy
import functools


from module.flow import cnf

import os
import cv2
import torch
import numpy as np
from models import ResnetGenerator
from utils import Preprocess
class Photo2Cartoon:
    def __init__(self):
        self.pre = Preprocess()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)

        assert os.path.exists(
            './models/ema_photo2cartoon_params_latest.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
        params = torch.load('./models/ema_photo2cartoon_params_latest.pt', map_location=self.device)
        self.net.load_state_dict(params['genA2B'])
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None

        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face * mask + (1 - mask) * 255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = torch.from_numpy(face).to(self.device)

        # inference
        with torch.no_grad():
            cartoon = self.net(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon


'''
    styleflow-webui demo
'''

DATA_ROOT = "./data"
HASH_FUNCS = {torch.nn.Module: id,
              torch.nn.parameter.Parameter: id,
              torch.Tensor: lambda x: x.cpu().numpy()}

# Select images
all_idx = np.array([2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362,
                             369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301,
                             599], dtype='int')

EPS = 1e-3  # arbitrary positive value

class State:  # Simple dirty hack for maintaining state
    prev_attr = None
    prev_idx = None
    first = True
    # ... and other state variables

if not hasattr(st, 'data'):  # Run only once. Save data globally

    st.state = State()
    with st.spinner("Setting up... This might take a few minutes"):
        raw_w = pickle.load(open(os.path.join(DATA_ROOT, "sg2latents.pickle"), "rb"))    # [39, 1, 18, 512]
        # raw_TSNE = np.load(os.path.join(DATA_ROOT, 'TSNE.npy'))  # We are picking images here by index instead
        raw_attr = np.load(os.path.join(DATA_ROOT, 'attributes.npy'))    # [39, 8, 1]
        raw_lights = np.load(os.path.join(DATA_ROOT, 'light.npy'))    # [39, 1, 9, 1, 1]

        all_w = np.array(raw_w['Latent'])[all_idx]
        all_attr = raw_attr[all_idx]
        all_lights = raw_lights[all_idx]

        light0 = torch.from_numpy(raw_lights[8]).float()
        light1 = torch.from_numpy(raw_lights[33]).float()
        light2 = torch.from_numpy(raw_lights[641]).float()
        light3 = torch.from_numpy(raw_lights[547]).float()
        light4 = torch.from_numpy(raw_lights[28]).float()
        light5 = torch.from_numpy(raw_lights[34]).float()

        pre_lighting = [light0, light1, light2, light3, light4, light5]

        st.data = dict(raw_w=raw_w, all_w=all_w, all_attr=all_attr, all_lights=all_lights,
                             pre_lighting=pre_lighting)

def np_copy(*args):  # shortcut to clone multiple arrays
    return [np.copy(arg) for arg in args]

@st.cache(allow_output_mutation=True, hash_funcs={dict: id}, show_spinner=False)
def get_idx2init(raw_w):
    print(type(raw_w))
    idx2init = {i: np.array(raw_w['Latent'])[i] for i in all_idx}
    return idx2init

# @st.cache
@st.cache(allow_output_mutation=True, hash_funcs=HASH_FUNCS)    # allow_output_mutation=True，忽略掉警告CachedObjectMutationWarning
def init_model():
    # 初始化stylegan2
    stylegan2_path = "./pretrained/ffhq.pkl"
    print("Loading stylegan2 networks from %s ..." % stylegan2_path)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with dnnlib.util.open_url(stylegan2_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    # print("stylegan2-G:", type(G))    # torch_utils.persistence.persistent_class.<locals>.Decorator
    # if torch.cuda.is_available() is False:
    #     G.forward = functools.partial(G.forward, force_fp32=True)    # cpu下强制用32位浮点数

    # 初始化styleflow
    styleflow_path = "./flow_weight/modellarge10k.pt"
    print("Loading styleflow networks from %s ..." % styleflow_path)
    prior = cnf(512, '512-512-512-512-512', 17, 1)
    prior.load_state_dict(torch.load(styleflow_path, map_location=torch.device('cpu')))
    prior.eval()

    # photo2cartoon初始化
    c2p = Photo2Cartoon()  # 2D捏脸

    # return prior.cpu()
    return G.float(), prior.cpu(), c2p

@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def flow_w_to_z(flow_model, w, attributes, lighting):
    w_cuda = torch.Tensor(w)
    att_cuda = torch.from_numpy(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    z = flow_model(w_cuda, features, zero_padding)[0].clone().detach()

    return z

@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def flow_z_to_w(flow_model, z, attributes, lighting):
    att_cuda = torch.Tensor(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    w = flow_model(z, features, zero_padding, True)[0].clone().detach().numpy()

    return w


def is_new_idx_set(idx):
    if st.state.first:
        st.state.first = False
        st.state.prev_idx = idx
        return True

    if idx != st.state.prev_idx:
        st.state.prev_idx = idx
        return True
    return False

def reset_state(idx):
    st.state = State()
    st.state.first = False
    st.state.prev_idx = idx

def make_slider(name, min_value=0.0, max_value=1.0, step=0.1, **kwargs):
    return st.sidebar.slider(name, min_value, max_value, step=step, **kwargs)

def get_changed_light(lights, light_names):
    for i, name in enumerate(light_names):
        change = abs(lights[name] - st.state.prev_lights[i])
        if change > EPS:
            return i
    return None

def preserve_w_id(w_new, w_orig, attr_index):
    # Ssssh! secret sauce to strip vectors
    w_orig = torch.Tensor(w_orig)
    if attr_index == 0:
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 1:
        w_new[0][:2] = w_orig[0][:2]
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 2:

        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 3:
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 4:
        w_new[0][6:] = w_orig[0][6:]

    elif attr_index == 5:
        w_new[0][:5] = w_orig[0][:5]
        w_new[0][10:] = w_orig[0][10:]

    elif attr_index == 6:
        w_new[0][0:4] = w_orig[0][0:4]
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 7:
        w_new[0][:4] = w_orig[0][:4]
        w_new[0][6:] = w_orig[0][6:]
    return w_new

@st.cache(show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def generate_image(model, w, noise_mode='const'):
    print("generate image with w %s ..." % str(w.shape))

    # # w要做成(1, 512)的
    # w = w[0, 0, :].reshape([-1, 512])
    # w = torch.Tensor(w)
    # # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)  # (1, 512)
    # label = torch.zeros([1, model.c_dim], device=device)
    #
    # img = model(w, label, truncation_psi=1, noise_mode='const')
    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)    # 做成NHWC
    # # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    # 这里使用style-mix的方式
    img = model.synthesis(torch.Tensor(w), noise_mode=noise_mode, force_fp32=True)  # w, [1, 18, 512]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return img[0].cpu().numpy().copy()

def main():
    # attribute_names = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']    # 8种属性
    attribute_names = ['性别', '戴眼镜', '旋转', '抬头', '脱发', '胡须', '年龄', '表情']    # 8种属性
    attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.]

    light_names = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

    # att_min = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': 0, 'Beard': 0.0, 'Age': 0,
    #            'Expression': 0}
    # att_max = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}
    att_min = {'性别': 0, '戴眼镜': 0, '旋转': -20, '抬头': -20, '脱发': 0, '胡须': 0.0, '年龄': 0,
               '表情': 0}
    att_max = {'性别': 1, '戴眼镜': 1, '旋转': 20, '抬头': 20, '脱发': 1, '胡须': 1, '年龄': 65, '表情': 1}


    with st.spinner("Setting up... This might take a few minutes... Please wait!"):
        all_w, all_attr, all_lights = np_copy(st.data["all_w"], st.data["all_attr"], st.data["all_lights"])
        pre_lighting = list(st.data["pre_lighting"])
        idx2w_init = get_idx2init(st.data["raw_w"])
        stylegan2_model, flow_model, c2p = init_model()
        # flow_model = init_model()

    # 这里选随机数，通过stylegan2生成图片
    idx_selected = st.selectbox("Choose an image:", list(range(len(idx2w_init))),
                                format_func=lambda opt: all_idx[opt])

    w_selected = all_w[idx_selected]    # [1, 18, 512]
    attr_selected = all_attr[idx_selected].ravel()    # [8, ]
    lights_selected = all_lights[idx_selected]    # [1, 9, 1, 1]
    z_selected = flow_w_to_z(flow_model, w_selected, attr_selected, lights_selected)    # [1, 18, 512]，原始空间映射到潜在空间

    if is_new_idx_set(idx_selected):
        reset_state(idx_selected)
        st.state.prev_attr = attr_selected.copy()
        st.state.prev_lights = lights_selected.ravel().copy()
        st.state.z_current = copy.deepcopy(z_selected)
        st.state.w_current = torch.Tensor(w_selected)
        st.state.w_prev = torch.Tensor(w_selected)
        st.state.light_current = torch.Tensor(lights_selected).float()

    st.sidebar.markdown("# Attributes")
    attributes = {}
    for i, att in enumerate(attribute_names):
        attributes[att] = make_slider(att, float(att_min[att]), float(att_max[att]),
                                      value=float(attr_selected.ravel()[i]),  # value on first render
                                      key=hash(idx_selected * 1e5 + i)  # re-render if index selected is changed!
                                      )

    st.sidebar.markdown("# Lighting")
    lights = {}
    for i, lt in enumerate(light_names):
        lights[lt] = make_slider(lt,
                                 value=float(lights_selected.ravel()[i]),  # value on first render
                                 key=hash(idx_selected * 1e6 + i)  # re-render if index selected is changed!
                                 )
    print("st.state.w_current:", type(st.state.w_current), st.state.w_current.shape)   # torch.Tensor, (1, 18, 512)

    img_source = generate_image(stylegan2_model, w_selected)    # 返回RGB的
    print("img_source.shape:", img_source.shape)

    # # 这里直接读取图片代替
    # import cv2
    # img_source = cv2.imread("0000.png")
    # img_source = cv2.resize(img_source, (1024, 1024))
    # img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)

    att_new = list(attributes.values())

    for i, att in enumerate(attribute_names):  # Not the greatest code, but works!
        attr_change = attributes[att] - st.state.prev_attr[i]

        if abs(attr_change) > EPS:
            print(f"Changed attr {att} : {attr_change}")
            attr_final = attr_degree_list[i] * attr_change + st.state.prev_attr[i]
            att_new[i] = attr_final
            print("\n")

            if hasattr(st.state, 'prev_changed') and st.state.prev_changed != att:
                st.state.z_current = flow_w_to_z(flow_model, st.state.w_current, st.state.prev_attr_factored,
                                                 lights_selected)
            st.state.prev_attr[i] = attributes[att]
            st.state.prev_changed = att
            st.state.prev_attr_factored = att_new
            st.state.w_current = flow_z_to_w(flow_model, st.state.z_current, att_new, lights_selected)
            break  # Streamlit re-runs on each interaction. Probably works but need to test for any bugs here

    pre_lighting_distance = [pre_lighting[i] - st.state.light_current for i in range(len(light_names))]
    lights_magnitude = np.zeros(len(light_names))
    changed_light_index = get_changed_light(lights, light_names)

    if changed_light_index is not None:
        lights_magnitude[changed_light_index] = lights[light_names[changed_light_index]]

        lighting_final = torch.Tensor(st.state.light_current)
        for i in range(len(light_names)):
            lighting_final += lights_magnitude[i] * pre_lighting_distance[i]

        w_current = flow_z_to_w(flow_model, st.state.z_current, att_new, lighting_final)

        w_current[0][0:7] = st.state.w_current[0][0:7]  # some stripping
        w_current[0][12:18] = st.state.w_current[0][12:18]

        st.state.w_current = w_current
        lights_new = lighting_final

        st.state.prev_lights[changed_light_index] = lights[light_names[changed_light_index]]
    else:
        lights_new = lights_selected

    col1, col2 = st.beta_columns(
        2)  # Columns feature of streamlit is still in beta. This line might require to be changed in future versions
    with col1:
        st.image(img_source, caption="Generated", use_column_width=True)

    with col2:
        st.state.w_current = preserve_w_id(st.state.w_current, st.state.w_prev, i)
        img_target = generate_image(stylegan2_model, st.state.w_current)    # 返回RGB的
        # img_target = np.ndarray([512, 512, 3])

        # img_target进入捏脸
        img_target = c2p.inference(img_target)    # 这里返回BGR的
        img_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)    # 再转回RGB的

        st.image(img_target, caption="Target", use_column_width=True)

    st.state.z_current = flow_w_to_z(flow_model, st.state.w_current, att_new, lights_new)
    st.state.w_prev = torch.Tensor(st.state.w_current).clone().detach()

if __name__ == '__main__':
    main()