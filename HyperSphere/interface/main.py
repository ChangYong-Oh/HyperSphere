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

from tasks.Rosenbrock import rosenbrock
from HyperSphere.interface.random_search import RandomSearch
from HyperSphere.interface.simple_GP import SimpleGP
import numpy as np

if __name__ == "__main__":
    task = rosenbrock
    ranges = [(-1, 3), (-1, 3)]
    random_search = RandomSearch(ranges)
    optimizer = SimpleGP(ranges)
    iterations = 200
    num_init = 25

    min_found = np.infty
    best_setting = None

    for i in range(num_init):
        setting = random_search.get_new_setting()
        result = task(setting)

        optimizer.submit_result(setting, result)
        print("Random setting {}, gives score: {}".format(setting, result))

    for i in range(iterations):
        setting = optimizer.get_new_setting()
        result = task(setting)

        optimizer.submit_result(setting, result)

        if result < min_found:
            min_found = result
            best_setting = setting

        print('iteration {}, current min found: {}'.format(i, min_found))

    print("Best minimum found: {}, at {}, "
          "after {} iterations".format(min_found, best_setting, iterations))

