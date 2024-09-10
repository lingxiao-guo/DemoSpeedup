import gym
import numpy as np
import cv2

class VideoWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            enabled=True,
            steps_per_render=1,
            **kwargs
        ):
        super().__init__(env)
        
        self.mode = mode
        self.enabled = enabled
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render

        self.frames = list()
        self.statelist = []
        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        if self.enabled:
            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.frames.append(frame)
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.statelist.append(result)
        self.step_count += 1
        self.statelist.append(result)
        
        if self.enabled and ((self.step_count % self.steps_per_render) == 0):
            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            frame = put_text(frame, f"{self.step_count}")
            self.frames.append(frame)
        # print('step count:', self.step_count)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        return self.frames
    

def put_text(img, text, is_waypoint=False, font_size=1, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img
