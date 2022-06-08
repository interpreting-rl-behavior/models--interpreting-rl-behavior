import gym
import numpy as np


class ResizeObservationVec(gym.wrappers.ResizeObservation):
    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "opencv is not install, run `pip install gym[other]`"
            )

        resized_imgs = []
        for img_idx in range(observation.shape[0]):
            resized_imgs.append(cv2.resize(
                observation[img_idx], self.shape[::-1], interpolation=cv2.INTER_AREA
            ))
        observation = np.stack(resized_imgs, axis=0)

        # observation = cv2.resize(
        #     observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        # )

        # if observation.ndim == 2:
        if observation.ndim == 3:
            observation = np.expand_dims(observation, -1)
        return observation

    def step_wait(self):
        obs, reward, done, info = self.env.step_wait()
        return self.observation(obs), reward, done, info