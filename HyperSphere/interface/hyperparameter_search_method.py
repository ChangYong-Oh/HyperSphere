#!/usr/bin/env python
# Created by Tijmen Blankevoort 2018 | tijmen@qti.qualcomm.com

#  ============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2018 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
#
#  ============================================================================

import numpy as np

class HyperParameterSearchMethod():
    def __init__(self, ranges):
        self.result_list = []
        self.experiment_number = 0
        self.ranges = ranges
        assert type(self.ranges) == list
        for tup in self.ranges:
            assert type(tup) == tuple or type(tup) == list

    def get_new_setting(self):
        raise NotImplementedError('This is a base method, not implemented')

    def submit_result(self, setting, result):
        self.result_list.append([setting,result])

    def get_best_found_setting(self):
        best_setting = None
        best_score = np.infty
        for pair in self.result_dict:
            if pair['result'] < best_score:
                best_score = pair['result']
                best_setting = pair['setting']

        return best_setting