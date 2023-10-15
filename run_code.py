# Copyright 2023, Karthik Ganapathy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import functionfile_speedygreedy as ff

if __name__ == "__main__":
    print('Code run start')

    exp_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # exp_list = [1, 9]
    # exp_list = [1]

    # run_flag = True
    run_flag = False

    plot_flag = True
    # plot_flag = False

    for exp_no in exp_list:

        ff.run_experiment(exp_no, run_check=run_flag, plot_check=plot_flag)

    print('Code run done')
