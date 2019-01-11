from ctypes import cdll, c_char_p, POINTER, c_int

lib = cdll.LoadLibrary('/home/ksenia/CLionProjects/KCFcpp/src/libtest.so')

lib.try_get_next_file.restype = c_char_p
lib.read_current_groundtruth.restype = POINTER(c_int)


class Tests:
    def __init__(self, path, prefix):
        lib.create_test_class(c_char_p(path.encode()),
                              c_char_p(prefix.encode()))

    def check_is_new_video(self):
        return lib.check_is_new_video()

    def try_get_next_file(self):
        result = lib.try_get_next_file()
        if not result:
            return None
        else:
            return result.decode()

    def bboxes_to_file(self, x1, y1, x2, y2):
        lib.bboxes_to_file(c_int(x1), c_int(y1), c_int(x2), c_int(y2))

    def time_to_file_init_time(self, time):
        lib.time_to_file_init_time(time)

    def time_to_file_track_time(self, time):
        lib.time_to_file_track_time(time)

    def read_current_groundtruth(self):
        return lib.read_current_groundtruth()[0:4]


# if __name__ == "__main__":
#     test = Tests("/media/ksenia/39e992da-a01a-4384-a580-e798bb2aab2a/datasets/cfnet-validation",
#                  "PY_KCF_")
#
#     path = test.try_get_next_file()
#
#     while path is not None:
#         # print(path)
#         path = test.try_get_next_file()
