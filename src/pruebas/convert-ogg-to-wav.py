import subprocess
src_filename = 'tos-test-0011.ogg'
dest_filename = 'tos-test-0011.wav'

process = subprocess.run(['c:\\dev\\ffmpeg\\bin\\ffmpeg', '-i', src_filename, dest_filename])
if process.returncode != 0:
    raise Exception("Something went wrong")