import sys
root = '/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4'
p = root + '/rl/native/ungroup.cpp'
s = open(p).read()
assert 'SEAT_SMART' not in s
s = s.replace("SEAT_KIDNAP = 5, SEAT_RAMMER = 6 };", "SEAT_KIDNAP = 5, SEAT_RAMMER = 6, SEAT_SMART = 7 };")
bot = open(root + '/../../../../' + 'tmp_dummy', 'w') if False else None
bot = open(sys.argv[1]).read()
s = s.replace("    void bot_action(int seat_type, int i, int* act) const {", bot + "\n    void bot_action(int seat_type, int i, int* act) const {")
s = s.replace("            case SEAT_RAMMER: bot_rammer(i, act); break;\n",
              "            case SEAT_RAMMER: bot_rammer(i, act); break;\n            case SEAT_SMART: bot_smart(i, act); break;\n")
open(p, 'w').write(s)
p = root + '/rl/ungroup/native.py'
s = open(p).read()
i = s.find('SEAT_EXTERNAL')
print(s[i:s.find('SEAT_LABEL')])
