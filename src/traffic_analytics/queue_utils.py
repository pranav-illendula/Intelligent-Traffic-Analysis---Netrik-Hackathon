def get_queue_metrics(tracked_boxes, speed_map, stop_line_y, frame_width):
    stopped_ids = set()
    occupied_area = 0

    for (x1, y1, x2, y2, vid) in tracked_boxes:
        speed = speed_map.get(vid, 999)

        # stopped vehicles before stop line
        if y2 < stop_line_y and speed < 3:
            stopped_ids.add(vid)
            occupied_area += (x2 - x1) * (y2 - y1)

    queue_length = len(stopped_ids)

    roi_area = stop_line_y * frame_width
    queue_density = min(occupied_area / roi_area, 1.0) if roi_area > 0 else 0

    return queue_length, queue_density
