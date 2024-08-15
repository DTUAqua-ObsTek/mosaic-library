import unittest
from mosaicking import utils
import cv2

class TestCUDAVideoPlayer(unittest.TestCase):
    def setUp(self):
        self._video = "../data/mosaicking/fishes.mp4"

    def test_full_playback(self):
        """
        Just read through the entire video and check that all frames are valid. Then try to read one more and check it is false.
        """
        player = utils.CUDAVideoPlayer(self._video)
        for i, (ret, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 2070)
        self.assertFalse(player.read()[0])
        del player

    def test_frameskip(self):
        """
        The video has 2070 frames, a frame skip of 120 should mean a read of 18 times.
        """
        player = utils.CUDAVideoPlayer(self._video, frame_skip=120)
        for i, (ret, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 18)
        self.assertFalse(player.read()[0])
        del player

    def test_start_position_int(self):
        """
        The video has 2070 frames and is 60 fps. If we start at frame 1800, there should be 270 reads left.
        """
        player = utils.CUDAVideoPlayer(self._video, start=1800)
        for i, (ret, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 270)
        self.assertFalse(player.read()[0])
        del player

    def test_stop_position_int(self):
        """
        Read up to and including frame 29 (zero indexed) There should only be 30 reads here.
        """
        player = utils.CUDAVideoPlayer(self._video, finish=29)
        for i, (ret, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 30)
        del player

    def test_stop_position_final(self):
        """
        Make sure VideoPlayers plays up to and including the specified frame.
        """
        player = utils.CUDAVideoPlayer(self._video, finish=2070)
        count = 0
        for i, (ret, frame) in enumerate(player):
            count = count + 1
            self.assertTrue(ret and frame is not None)
        self.assertEqual(count, 2070)
        del player

    def test_start_stop(self):
        """
        Player starts at 2060 and finishes after 2069 (zero indexed). Should be 10 frames read.
        """
        player = utils.CUDAVideoPlayer(self._video, start=2060, finish=2070)
        for i, (ret, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 10)
        del player

    def test_start_time(self):
        """
        Player should read 1 complete second of frames which will include the start of the next second (i==61)
        """
        player = utils.CUDAVideoPlayer(self._video, start="00:00:15", finish="00:00:16")
        for i, (ret, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        player.get(cv2.CAP_PROP_POS_MSEC)
        self.assertEqual(i, 61)
        del player


if __name__ == '__main__':
    unittest.main()
