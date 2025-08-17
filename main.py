from workflow import run_workflow

def main():
    import time
    questions = [
        "Estimate crop yield for wheat in Punjab in winter of 2025",
        "How can farmers manage pest outbreaks in cotton fields?",
        "What is the market price trend for wheat in India?",
        "How to prevent fungal diseases in tomato crops?",
    ]
    image_queries = [
        ("Analyze this crop disease", "Images/Crop/crop_disease.jpg"),
        ("Check for pests in this image", "Images/Pests/jpg_0.jpg"),
    ]

    mode = input("Select initial mode (rag/tooling): ").strip().lower()
    if mode not in ["rag", "tooling"]:
        print("Invalid mode. Using 'rag' as default.")
        mode = "rag"

    query_type = input("Test type (text/image): ").strip().lower()
    if query_type == "image":
        test_queries = image_queries
    else:
        test_queries = [(q, None) for q in questions]

    for idx, (user_query, image_path) in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}: {user_query}")
        if image_path:
            print(f"Image Path: {image_path}")
        print(f"Initial Mode: {mode.upper()}")
        print('='*80)

        start_time = time.time()
        result = run_workflow(user_query, mode, image_path)
        end_time = time.time()

        print(f"Answer: {result['answer']}")
        print(f"\nQuality Metrics:")
        print(f"  - Is Answer Complete: {result['is_answer_complete']}")
        print(f"  - Final Mode: {result['final_mode']}")
        print(f"  - Switched Modes: {result['switched_modes']}")
        print(f"  - Is Image Query: {result['is_image_query']}")
        print(f"  - Processing Time: {end_time - start_time:.2f}s")

        if result['answer_quality_grade'].get('reasoning'):
            print(f"  - Quality Grade Reasoning: {result['answer_quality_grade']['reasoning']}")

if __name__ == "__main__":
    main()
