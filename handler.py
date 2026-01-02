import runpod

def handler(job):
    """
    Mock RunPod Worker.
    Simply receives the file (base64) and returns it as a mock 'cleansed' version.
    """
    job_input = job["input"]
    task = job_input.get("task")
    
    if task == "video_removal":
        # Mock: Return the same video data passed in
        video_data = job_input.get("video")
        return {"cleansed_video": video_data, "status": "mocked_success"}
    
    return {"error": "Invalid task"}

runpod.serverless.start({"handler": handler})
