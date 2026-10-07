#!/bin/bash
# VTD 검증 — 시나리오마다 선생님·학생이 한 판씩 달린다. VTD PC 에서 돌린다(데스크톱 세션이 로그인돼 있어야 한다:
# 규칙 스택 vtd/load_scenario.py 가 VTD 화면을 xdotool 로 조작한다).
#   사용: bash scripts/vtd_validate.sh <출력폴더> <plain|ev> [코스 글자들(기본 A B D E G H)]
#   plain = HL_FMA_NEW_<코스>.xml(신호만, 단계 ② 에 해당), ev = HL_FMA_NEW_<코스>_EV.xml(내 차로 정지차 + 대향차)
# 환경변수: CKPT(학생 그물), HLFMA(규칙 스택 작업본, 기본 ~/hlfma2026), VTD_ROOT(기본 ~/Hexagon/VTD.2025.2), DISPLAY(기본 :1)
cd "$(dirname "$0")/.." || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
HLFMA=${HLFMA:-$HOME/hlfma2026}; VTD_ROOT=${VTD_ROOT:-$HOME/Hexagon/VTD.2025.2}
PY="env -u PYTHONPATH .venv/bin/python"
O=${1:?출력폴더}; KIND=${2:?plain|ev}; shift 2
COURSES=${*:-A B D E G H}
CK=${CKPT:?CKPT=<학생 그물 .pt> 를 준다}
mkdir -p $O

alive() { pgrep -x taskControl >/dev/null && pgrep ghostdriver >/dev/null; }
scp_cmd() { python3 -c "import socket,struct,sys
d=sys.argv[1].encode();h=struct.pack('<HH64s64sI',40108,1,b'cmd',b'TaskControl',len(d))
s=socket.create_connection(('127.0.0.1',48179),timeout=5);s.sendall(h+d);s.close()" "$1"; }
restart_vtd() {
  echo "  VTD 재시작 $(date +%H:%M:%S)"
  "$VTD_ROOT/bin/vtdStop.sh" >/dev/null 2>&1; sleep 3
  pkill -x moduleManager; pkill -9 -f "$VTD_ROOT/Runtime"; pkill -f startSlave.sh; sleep 4
  (cd "$VTD_ROOT" && setsid ./bin/vtdStart.sh </dev/null >/tmp/vtdauto.log 2>&1 &)
  sleep 85
  for _ in 1 2 3; do alive && return 0; sleep 20; done
  return 1
}

echo "=== VTD 검증 시작 $(date +%H:%M:%S) 종류=$KIND 코스=$COURSES ==="
for c in $COURSES; do
  NAME=HL_FMA_NEW_$c; [ "$KIND" = ev ] && NAME=${NAME}_EV
  for WHO in teacher student; do
    R=$O/$NAME-$WHO
    if [ -f $R/result.json ]; then echo "$NAME $WHO 이미 있음 — 건너뜀"; continue; fi
    alive || restart_vtd || { echo "❌ VTD 를 못 띄웠다"; exit 1; }
    loaded=0
    for try in 1 2 3; do
      if (cd "$HLFMA/vtd" && timeout 150 python3 load_scenario.py $NAME) >> $R.load.log 2>&1; then loaded=1; break; fi
      echo "  $NAME 로드 실패(${try}차)"; tail -2 $R.load.log; sleep 30
      [ $try = 2 ] && restart_vtd
    done
    [ $loaded = 1 ] || { echo "❌ $NAME 로드 실패 — 건너뜀"; continue; }
    if [ $WHO = teacher ]; then A="--teacher"; else A="--ckpt $CK --lim-anticipate"; fi
    echo "▶ $NAME $WHO $(date +%H:%M:%S)"
    $PY scripts/drive_vtd.py --board course_$c --curriculum curricula/stage2.json $A --out $R > $R.log 2>&1 < /dev/null
    tail -1 $R.log
    scp_cmd '<SimCtrl><Stop/></SimCtrl>' 2>/dev/null; sleep 3
  done
done
echo "=== VTD 검증 끝 $(date +%H:%M:%S) ==="
