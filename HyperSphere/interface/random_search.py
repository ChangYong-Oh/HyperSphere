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

from HyperSphere.interface.hyperparameter_search_method import HyperParameterSearchMethod

import numpy as np


class RandomSearch(HyperParameterSearchMethod):
    def __init__(self, *args):
        super(RandomSearch, self).__init__(*args)

    def get_new_setting(self):
        x = np.empty(len(self.ranges))

        i = 0
        for param_range in self.ranges:
            x[i] = np.random.uniform(*param_range)
            i += 1

        return x