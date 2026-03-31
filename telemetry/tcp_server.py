from telemetry.stream import ml_queue

def handle_packet(packet):
    parsed = parse_packet(packet)

    # Push to queue instead of direct ML call
    ml_queue.put(parsed)