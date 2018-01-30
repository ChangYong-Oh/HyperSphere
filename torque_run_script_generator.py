import os
import sys


HEAD = ""
HEAD += "#Set job requirements\n"
HEAD += "#PBS -S /bin/bash\n"
HEAD += "#PBS -lnodes=1:ppn=16\n"
HEAD += "#PBS -lwalltime=1:30:00\n"
HEAD += "\n"
HEAD += "#Loading modules\n"
HEAD += "module load python\n"
HEAD += "\n"

BODY_INITIATING = ""
BODY_INITIATING += "INIT_COMMAND_CUBE_ROOT = \"python HyperSphere/BO/run_BO.py --parallel -g cube\"\n"
BODY_INITIATING += "INIT_COMMAND_SPHERE_ROOT = \"python HyperSphere/BO/run_BO.py --parallel -g sphere --origin\"\n"
BODY_INITIATING += "case \"$ALGORITHM\" in\n"
BODY_INITIATING += "  cube) INIT_COMMAND_PREFIX = \"$INIT_COMMAND_CUBE_ROOT\"\n"
BODY_INITIATING += "    ;;\n"
BODY_INITIATING += "  cubeard) INIT_COMMAND_PREFIX = \"$INIT_COMMAND_CUBE_ROOT --ard\"\n"
BODY_INITIATING += "    ;;\n"
BODY_INITIATING += "  none) INIT_COMMAND_PREFIX = \"$INIT_COMMAND_SPHERE_ROOT\"\n"
BODY_INITIATING += "    ;;\n"
BODY_INITIATING += "  boundary) INIT_COMMAND_PREFIX = \"$INIT_COMMAND_SPHERE_ROOT --boundary\"\n"
BODY_INITIATING += "    ;;\n"
BODY_INITIATING += "  warping) INIT_COMMAND_PREFIX = \"$INIT_COMMAND_SPHERE_ROOT --warping\"\n"
BODY_INITIATING += "    ;;\n"
BODY_INITIATING += "  warpingboundary) INIT_COMMAND_PREFIX = \"$INIT_COMMAND_SPHERE_ROOT --warping --boundary\"\n"
BODY_INITIATING += "    ;;\n"
BODY_INITIATING += "esac\n"
BODY_INITIATING += "\n"
BODY_INITIATING += "INIT_COMMAND=\"$INIT_COMMAND_PREFIX -f $TASK -d $DIM\"\n"
BODY_INITIATING += "EVALSTDOUT=$(eval \"$INIT_COMMAND\")\n"
BODY_INITIATING += "EXPPATH=\"${EVALSTDOUT##*$'\\n'}\"\n"

BODY_CONTINUING = ""
BODY_CONTINUING += "CONTINUED_COMMAND=\"python HyperSphere/BO/run_BO.py -p $EXPPATH\"\n"
BODY_CONTINUING += "for ((i=1;i<=$NEVAL;i++)); do\n"
BODY_CONTINUING += "  eval \"$CONTINUED_COMMAND\"\n"
BODY_CONTINUING += "done\n"


def generate_initiating_script(algorithm, func_name, dim, n_eval):
	script = HEAD
	script += "ALGORITHM=" + algorithm + "\n"
	script += "TASK=" + func_name + "\n"
	script += "DIM=" + str(dim) + "\n"
	script += "NEVAL=" + str(n_eval) + "\n"
	script += "\n"
	script += BODY_INITIATING
	script += BODY_CONTINUING
	print(script)


def generate_continuing_script(pathname):
	script = HEAD
	script += "EXPPATH=" + os.path.realpath(pathname) + "\n"
	script += "\n"
	script += BODY_CONTINUING
	print(script)


if __name__ == '__main__':
	if len(sys.argv) == 5:
		generate_initiating_script(*sys.argv[1:])
	elif len(sys.argv) == 2:
		generate_continuing_script(sys.argv[1])