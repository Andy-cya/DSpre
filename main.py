import PIL.Image
import streamlit as st
import random
import base64
import io
import imageio.v2 as imageio
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
import time
from env import *
import torch
import time
from PIL import Image

def generate_gif(length, width, height, n_wall, startend):
    _lock = RendererAgg.lock
    with _lock:
        # 制作gif图
        start_time = time.time()
        pics1 = []
        pics2 = []
        pics3 = []

        LEARNING_RATE = 0.00033  # 学习率
        num_episodes = 80000  # 训练周期长度
        space_dim = 140  # n_spaces   状态空间维度
        action_dim = 7  # n_actions   动作空间维度
        threshold = 200
        env = Env(space_dim, action_dim, LEARNING_RATE)

        check_point_Qlocal = torch.load('Qlocal.pth', map_location=torch.device('cpu'))
        check_point_Qtarget = torch.load('Qtarget.pth', map_location=torch.device('cpu'))
        env.q_target.load_state_dict(check_point_Qtarget['model'])
        env.q_local.load_state_dict(check_point_Qlocal['model'])
        env.optim.load_state_dict(check_point_Qlocal['optimizer'])
        epoch = check_point_Qlocal['epoch']
        # 真实场景运行
        env.type = 1
        env.len = length
        env.width = width
        env.h = height
        env.n_wall = n_wall
        env.startend = startend


        state = env.reset_test()  # 环境重置1
        total_reward = 0
        env.render(1)
        n_done = 0
        count = 0

        n_test = 1  # 测试次数
        n_creash = 0  # 坠毁数目
        success_count = 0

        step = 0

        while 1:
            if env.agents[0].done:
                # 无人机已结束任务，跳过
                break
            action = env.get_action(FloatTensor(np.array([state[0]])), 0)  # 根据Q值选取动作

            next_state, reward, uav_done, info = env.step(action.item(), 0)  # 根据选取的动作改变状态，获取收益

            total_reward += reward  # 求总收益
            # 交互显示
            # print(action)
            env.render()
            # plt.pause(0.01)

            if step % int(env.agents[0].d_origin / 5) == 0:
                env.ax.view_init(elev=45, azim=45)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                pics1.append(imageio.imread(buffer))
                buffer.close()
                env.ax.view_init(elev=45, azim=180)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                pics2.append(imageio.imread(buffer))
                buffer.close()
                env.ax.view_init(elev=-90, azim=0)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                pics3.append(imageio.imread(buffer))
                buffer.close()

            if uav_done:
                print(info)
                break
            if info == 1:
                success_count = success_count + 1

            step += 1

            state[0] = next_state  # 状态变更

        # print(env.agents[0].step)
        # print(env.state)
        # print(env.agents[0].distance)
        env.ax.scatter(env.target[0].x, env.target[0].y, env.target[0].z, c='red')
        env.ax.view_init(elev=45, azim=45)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        pics1.append(imageio.imread(buffer))
        buffer.close()
        env.ax.view_init(elev=45, azim=180)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        pics2.append(imageio.imread(buffer))
        buffer.close()
        env.ax.view_init(elev=-90, azim=0)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        pics3.append(imageio.imread(buffer))
        buffer.close()

        # plt.pause(5)
        buffer = io.BytesIO()
        imageio.mimsave(buffer, pics1, duration=0.01, format='gif')
        gif1 = imageio.mimread(buffer)
        buffer.close()
        buffer = io.BytesIO()
        imageio.mimsave(buffer, pics2, duration=0.01, format='gif')
        gif2 = imageio.mimread(buffer)
        buffer.close()
        buffer = io.BytesIO()
        imageio.mimsave(buffer, pics3, duration=0.01, format='gif')
        gif3 = imageio.mimread(buffer)
        buffer.close()

        buffer = io.BytesIO()
        imageio.mimsave(buffer, gif1, fps=10, format='gif')
        gif1 = buffer.getvalue()
        buffer.close()
        buffer = io.BytesIO()
        imageio.mimsave(buffer, gif2, fps=10, format='gif')
        gif2 = buffer.getvalue()
        buffer.close()
        buffer = io.BytesIO()
        imageio.mimsave(buffer, gif3, fps=10, format='gif')
        gif3 = buffer.getvalue()
        buffer.close()

        end_time = time.time()  # 程序结束时间
        run_time = end_time - start_time  # 程序的运行时间，单位为秒
        print(run_time)
        return gif1, gif2, gif3


gif1 = False

st.set_page_config(page_title="automated pipe layout")

with st.sidebar:
    # st.text("请选择房间类型\n")
    # type=st.selectbox(
    #     "none",
    #     ('普通房间','机房'),
    #     label_visibility="collapsed"
    #     )

    # st.button("随机生成房间尺寸")
    # st.slider("长", 5.0, 15.0)
    # st.slider("长宽比", 1.0, 1.4)
    # st.slider("高", 2.5, 5.0)

    st.title("随机生成\n")
    st.text("将随机生成房间尺寸，起止点位置，隔墙数目")
    if st.button("确认随机生成"):
        length = random.randrange(100, 300)
        width = int(length * random.uniform(0.7, 1))
        height = int(max(length * random.uniform(0.3, 0.4), 50))
        if length < 200:
            n_wall = 0  # 原来为0
        else:
            n_wall = random.randint(1, 2)
        startend = 0
        gif1, gif2, gif3 = generate_gif(length, width, height, n_wall, startend)



    st.title("自定义\n")
    length = int(st.slider("长(m)", 5.0, 15.0) * 20)
    width = int(length / st.slider("长宽比", 1.0, 1.4))
    height = int(st.slider("高(m)", 2.5, 5.0) * 20)
    n_wall = st.slider("隔墙数量", 0, 2)
    str = st.select_slider('起止点相对关系', options=['相邻墙上', '相对墙上'])
    if str == '相邻墙上':
        startend = 1
    if str == '相对墙上':
        startend = 2
    if st.button("完成自定义"):
        gif1, gif2, gif3 = generate_gif(length, width, height, n_wall, startend)
        

    st.title("查询演示\n")
    a = st.slider("隔墙数量：", 0, 2)
    b = st.select_slider('起止点相对关系：', options=['相邻墙上', '相对墙上'])
    if st.button("查询演示"):
        if a == 0 and b == '相邻墙上':
            st.image(
                "https://imgse.com/i/zX6qBV",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/Hn13jW0T/15-00-15-00-5-00-0-0-2.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/VNbjMQkv/15-00-15-00-5-00-0-0-3.gif",
            width=400,
            )       
            st.text('栅格大小为0.05m')
        if a == 0 and b == '相对墙上':
            st.image(
                "https://i.postimg.cc/fLnYf2VK/15-00-15-00-5-00-0-1-1.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/TwVg7VmX/15-00-15-00-5-00-0-1-2.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/mkTM0PN1/15-00-15-00-5-00-0-1-3.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')        
        if a == 1 and b == '相邻墙上':
            st.image(
                "https://i.postimg.cc/fTTmC0R6/15-00-15-00-5-00-1-0-1.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/Wp7007TG/15-00-15-00-5-00-1-0-2.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/RhGf9vXs/15-00-15-00-5-00-1-0-3.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
        if a == 1 and b == '相对墙上':    
            st.image(
                "https://i.postimg.cc/rFnWPFdz/15-00-15-00-5-00-1-1-1.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/NjKTqqTD/15-00-15-00-5-00-1-1-2.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/Qdv7j4J3/15-00-15-00-5-00-1-1-3.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
        if a == 2 and b == '相邻墙上':
            st.image(
                "https://i.postimg.cc/4dgh9Y02/15-00-15-00-5-00-2-0-1.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/9MYwtW5m/15-00-15-00-5-00-2-0-2.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/prVhsgRD/15-00-15-00-5-00-2-0-3.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
        if a == 2 and b == '相对墙上':
            st.image(
                "https://i.postimg.cc/90x4KJQs/15-00-15-00-5-00-2-1-1.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/KcQg1vDQ/15-00-15-00-5-00-2-1-2.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')
            st.image(
                "https://i.postimg.cc/Hsx8XDSm/15-00-15-00-5-00-2-1-3.gif",
            width=400,
            )
            st.text('栅格大小为0.05m')


        
        
if gif1:
    st.image(gif1, use_column_width='auto')
    st.text('栅格大小为0.05m')
    st.image(gif2, use_column_width='auto')
    st.text('栅格大小为0.05m')
    st.image(gif3, use_column_width='auto')
    st.text('栅格大小为0.05m')

