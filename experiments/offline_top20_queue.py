"""One-shot local dependency queue; standard library only, no network or push."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiment_io import atomic_json
from research_gpu_guard import check_gpu_processes

def dependency_ready(status, processes_alive):
    if status.get('status') in ['failed','failed_or_requires_review']:
        raise RuntimeError('Preceding experiment failed; stopped for review')
    return status.get('status')=='complete' and not processes_alive

def alive(pid):
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        if kernel.GetLastError()==5: return True
        return False
    code=ctypes.c_ulong()
    try:
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code)):
            raise RuntimeError('Cannot inspect preceding process')
        return code.value==259
    finally: kernel.CloseHandle(ctypes.c_void_p(handle))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('config');args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    out=Path(cfg['queue_directory']);root=Path(cfg['root'])
    with (out/'queue.lock').open('x') as f:f.write(str(os.getpid()))
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))
    deadline=time.monotonic()+48*3600;idle=0
    def record(status,**kw):atomic_json(out/'status.json',dict(status=status,pid=os.getpid(),time=time.time(),**kw))
    try:
        while time.monotonic()<deadline:
            if (out/'STOP').exists(): record('cancelled_before_launch');return 0
            state={'status':'waiting_for_process_exit' if any(alive(pid) for pid in cfg['dependency_pids']) else 'complete'}
            ready=dependency_ready(state,any(alive(pid) for pid in cfg['dependency_pids']))
            idle_now=False
            if ready:
                try:
                    check_gpu_processes()
                    q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                    memory,util=map(int,q.strip().split(',')[:2]);idle_now=memory<512 and util<10
                except (RuntimeError,subprocess.SubprocessError,ValueError): idle_now=False
            idle=idle+1 if idle_now else 0
            record('waiting',dependency_status=state.get('status'),dependency_ready=ready,idle_checks=idle)
            if idle>=3: break
            time.sleep(30)
        else: raise RuntimeError('48-hour waiting limit reached; no training launched')
        current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
        if current!=cfg['commit'] or branch!=cfg['branch'] or dirty: raise RuntimeError('Checkout changed while queued')
        check_gpu_processes()
        record('launching')
        env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',TEMP=str(out/'tmp'),TMP=str(out/'tmp'))
        with (out/'training.log').open('xb') as log:
            child=subprocess.Popen([cfg['python'],'-B','-u',str(root/cfg['entrypoint'])],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            record('training',child_pid=child.pid)
            rc=child.wait()
        record('completed' if rc==0 else 'training_failed',exit_code=rc)
        return rc
    except BaseException:
        record('stopped_requires_review',traceback=traceback.format_exc());raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000))

if __name__=='__main__':raise SystemExit(main())
