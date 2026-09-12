"""Replace the bot_smart block (from '    // Smart:' to '    void bot_action') with the given file, then rebuild."""
import subprocess
import sys

root = '/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4'
p = root + '/rl/native/ungroup.cpp'
s = open(p).read()
a = s.find('    // Smart:')
b = s.find('    void bot_action')
assert 0 < a < b
new = open(sys.argv[1]).read()
if not new.endswith('\n'):
    new += '\n'
s = s[:a] + new.lstrip('\n') + '\n' + s[b:]
open(p, 'w').write(s)
subprocess.check_call(['g++', '-O3', '-march=native', '-std=c++17', '-fopenmp', '-shared', '-fPIC'] + sys.argv[2:] + ['-o', 'libungroup.so', 'ungroup.cpp'],
                      cwd=root + '/rl/native')
print('built', sys.argv[1])
