cd / && python /tmp/dump_chapters.py > /tmp/chapters_dump.txt 2>&1; echo "EXIT: $?"; wc -l /tmp/chapters_dump.txt
