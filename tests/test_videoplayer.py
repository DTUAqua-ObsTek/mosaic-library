import unittest
from mosaicking import utils
import cv2
import mosaicking
from pathlib import Path

class TestVideoPlayer(unittest.TestCase):
    def setUp(self):
        self._video = self._video = Path(__file__).parent / 'data' / "sparus_snippet.mp4"

    def test_full_playback(self):
        """
        Just read through the entire video and check that all frames are valid. Then try to read one more and check it is false.
        """
        player = utils.CPUVideoPlayer(self._video)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertFalse(player.read()[0])
        player.release()

    def test_frameskip(self):
        """
        The video has 22 frames, a frame skip of 2 should mean a read of 11 times.
        """
        player = utils.CPUVideoPlayer(self._video, frame_skip=2)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 11)
        self.assertFalse(player.read()[0])
        player.release()

    def test_start_position_int(self):
        """
        The video has 22 frames and is 5 fps. If we start at frame 15, there should be 7 reads left.
        """
        player = utils.CPUVideoPlayer(self._video, start=15)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 7)
        self.assertFalse(player.read()[0])
        player.release()

    def test_stop_position_int(self):
        """
        Read up to and including frame 14 (zero indexed) There should only be 15 reads here.
        """
        player = utils.CPUVideoPlayer(self._video, finish=14)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 15)
        player.release()

    def test_stop_position_final(self):
        """
        Make sure VideoPlayers plays up to and including the specified frame.
        """
        player = utils.CPUVideoPlayer(self._video, finish=18)
        for i, (ret, frame_no, name, frame) in enumerate(player):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 18)
        player.release()

    def test_start_stop(self):
        """
        Player starts at 10 and finishes after 15 (zero indexed). Should be 5 frames read.
        """
        player = utils.CPUVideoPlayer(self._video, start=10, finish=15)
        for i, (ret, frame_no, name, frame) in enumerate(player):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 5)
        player.release()

    def test_start_time(self):
        """
        Player should read 1 complete second of frames which will include the start of the next second (i==6)
        """
        player = utils.CPUVideoPlayer(self._video, start="00:00:01", finish="00:00:02")
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        player.get(cv2.CAP_PROP_POS_MSEC)
        self.assertEqual(i, 6)
        player.release()


@unittest.skipIf(not mosaicking.HAS_CODEC, "Skipping CUDAVideoPlayer tests: cv2.cudacodec is not available.")
class TestCUDAVideoPlayer(unittest.TestCase):
    def setUp(self):
        self._video = Path(__file__).parent / 'data' / "aqualoc_snippet.mp4"

    def test_full_playback(self):
        """
        Just read through the entire video and check that all frames are valid. Then try to read one more and check it is false.
        """
        player = utils.CUDAVideoPlayer(self._video)
        for i, (ret, frame_no, name, frame) in enumerate(player):
            self.assertTrue(ret and frame is not None)
        self.assertFalse(player.read()[0])
        del player

    def test_frameskip(self):
        """
        The video has 80 frames, a frame skip of 8 should mean a read of 10 times.
        """
        player = utils.CUDAVideoPlayer(self._video, frame_skip=8)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 10)
        self.assertFalse(player.read()[0])
        del player

    def test_start_position_int(self):
        """
        The video has 80 frames and is 5 fps. If we start at frame 13, there should be 67 reads left.
        """
        player = utils.CUDAVideoPlayer(self._video, start=13)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 67)
        self.assertFalse(player.read()[0])
        del player

    def test_stop_position_int(self):
        """
        Read up to and including frame 13 (zero indexed) There should only be 14 reads here.
        """
        player = utils.CUDAVideoPlayer(self._video, finish=13)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 14)
        del player

    def test_stop_position_final(self):
        """
        Make sure VideoPlayers plays up to and including the specified frame.
        """
        player = utils.CUDAVideoPlayer(self._video, finish=19)
        for i, (ret, frame_no, name, frame) in enumerate(player):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 19)
        del player

    def test_start_stop(self):
        """
        Player starts at 60 and finishes after 70 (zero indexed). Should be 11 frames read.
        """
        player = utils.CUDAVideoPlayer(self._video, start=60, finish=70)
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        self.assertEqual(i, 11)
        del player

    def test_start_time(self):
        """
        Player should read 1 complete second of frames which will include the start of the next second (i==21)
        """
        player = utils.CUDAVideoPlayer(self._video, start="00:00:02", finish="00:00:03")
        for i, (ret, frame_no, name, frame) in enumerate(player, start=1):
            self.assertTrue(ret and frame is not None)
        player.get(cv2.CAP_PROP_POS_MSEC)
        self.assertEqual(i, 21)
        del player



if __name__ == '__main__':
    unittest.main()
