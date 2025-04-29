from filelock import FileLock
import sys

queue_fp = "/nfs/stak/users/brownhun/ckpt_tracker.txt"
#queue_fp = "/home/hunter/output.txt"
tmp_file = "/nfs/stak/users/brownhun/next_ckpt_holder.txt"
#tmp_file = "/home/hunter/tmp_out_infile.txt"
if __name__=="__main__":
    if len(sys.argv) > 1:
        tmp_file = sys.argv[1]
    output = ""
    lock = FileLock(queue_fp + ".lock")
    with lock:
        print("got lock")
        raw_args = None
        with open(queue_fp, "r+") as f:
            raw_args = f.readline().strip().split()
            data = f.read()
            f.truncate(0)
            f.write(data)
        print("got new data")
        output = f"{raw_args[0]} {raw_args[1]} {raw_args[2]}"
        with open(tmp_file, 'w') as f:
            f.write(output)
        print("wrote new data")
    sys.exit(0)
            
