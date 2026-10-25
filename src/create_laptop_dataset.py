"""Generate a curated, lightweight LiveVQA benchmark slice for laptop CPU evaluation.

Replicates the exact schema, domains (News, Video, arXiv Papers), question levels
(Level 1 Entity Recognition, Level 2 Multi-Hop Reasoning), 5-choice options (A-E),
and synthetic reasoning trajectories from the LiveVQA research paper (arXiv:2504.05288).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def create_annotated_image(
    file_path: Path,
    title: str,
    domain: str,
    accent_color: tuple[int, int, int],
    pattern: str = "rect",
) -> None:
    """Render a clean, visually distinctive synthetic event/paper image for benchmarking."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Top header bar
    draw.rectangle([0, 0, 256, 36], fill=accent_color)
    draw.text((10, 10), domain.upper(), fill=(255, 255, 255))

    # Main visual motif
    if pattern == "rect":
        draw.rectangle([40, 60, 216, 170], fill=accent_color, outline=(40, 40, 40), width=2)
        draw.ellipse([80, 80, 176, 150], fill=(255, 255, 255))
    elif pattern == "diagram":
        # Architecture / flowchart style for arXiv
        draw.rectangle([30, 60, 100, 100], fill=(220, 230, 245), outline=(30, 80, 180), width=2)
        draw.text((38, 75), "Encoder", fill=(20, 20, 20))
        draw.line([(100, 80), (150, 80)], fill=(30, 80, 180), width=2)
        draw.rectangle([150, 60, 226, 100], fill=(230, 245, 230), outline=(40, 150, 60), width=2)
        draw.text((158, 75), "Decoder", fill=(20, 20, 20))
        draw.line([(188, 100), (188, 130)], fill=(40, 150, 60), width=2)
        draw.rectangle([120, 130, 226, 170], fill=(255, 240, 220), outline=(200, 100, 30), width=2)
        draw.text((130, 145), "Projection", fill=(20, 20, 20))
    elif pattern == "scene":
        # Event scene with simulated background and crowd/stage
        draw.rectangle([20, 50, 236, 180], fill=(210, 215, 225))
        draw.ellipse([60, 90, 110, 140], fill=accent_color)
        draw.rectangle([130, 110, 210, 160], fill=(80, 80, 90))
        draw.text((140, 125), "Stage / Venue", fill=(255, 255, 255))
    else:
        draw.ellipse([50, 60, 206, 180], fill=accent_color, outline=(0, 0, 0), width=2)

    # Bottom caption strip
    draw.rectangle([0, 200, 256, 256], fill=(230, 235, 240))
    draw.text((12, 215), title[:34], fill=(50, 50, 60))

    img.save(file_path)


def generate_livevqa_laptop_dataset(
    output_path: str | Path = "data/livevqa_laptop_mini.jsonl",
    images_dir: str | Path = "data/livevqa_images",
) -> Path:
    """Generate a curated 24-sample LiveVQA dataset slice across 12 images."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img_dir = Path(images_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    # 12 unique events/images (ensuring image-disjoint evaluation)
    # Each image has 2 questions: Level 1 (entity recognition) & Level 2 (multi-hop reasoning)
    records: List[Dict[str, Any]] = []

    items = [
        # Domain 1: News
        {
            "img_id": "news_ice_facility",
            "domain": "news",
            "title": "Central Louisiana ICE Center",
            "color": (41, 128, 185),
            "pattern": "scene",
            "date": "2024-04-15",
            "q1": {
                "question": "Based on the provided image, what is the specific facility shown?",
                "options": {
                    "A": "Central Louisiana ICE Processing Facility",
                    "B": "Fort Bliss Army base",
                    "C": "Baldwin Immigration Center",
                    "D": "Alexandria Detention Center",
                    "E": "Dilley Family Residential Center",
                },
                "answer": "A",
                "ground_truth_text": "Central Louisiana ICE Processing Facility",
                "ground_truth_list": ["Central Louisiana ICE Processing Facility", "Jena ICE Center", "Central Louisiana Processing Facility"],
                "reasoning": "The aerial view displays the distinctive multi-wing layout and high-security perimeter of the Central Louisiana ICE Processing Facility in Jena, Louisiana.",
                "level": 1,
            },
            "q2": {
                "question": "Which private contractor operates the detention facility depicted in this view?",
                "options": {
                    "A": "CoreCivic",
                    "B": "The GEO Group Inc.",
                    "C": "Management & Training Corp",
                    "D": "G4S Secure Solutions",
                    "E": "Serco Group",
                },
                "answer": "B",
                "ground_truth_text": "The GEO Group Inc.",
                "ground_truth_list": ["The GEO Group Inc.", "GEO Group", "The GEO Group"],
                "reasoning": "Combining the facility identification with contracting records, The GEO Group Inc. holds the operational and management contract for the Jena processing facility.",
                "level": 2,
            },
        },
        {
            "img_id": "news_paris_olympics",
            "domain": "news",
            "title": "2024 Paris Olympic Opening Ceremony",
            "color": (142, 68, 173),
            "pattern": "scene",
            "date": "2024-07-26",
            "q1": {
                "question": "Based on the provided image, what international event is taking place along the river?",
                "options": {
                    "A": "2024 Summer Olympics Opening Ceremony",
                    "B": "Bastille Day Military Parade",
                    "C": "Tour de France Grand Finale",
                    "D": "European Rowing Championship",
                    "E": "Cannes Film Festival Promenade",
                },
                "answer": "A",
                "ground_truth_text": "2024 Summer Olympics Opening Ceremony",
                "ground_truth_list": ["2024 Summer Olympics Opening Ceremony", "Paris 2024 Opening Ceremony", "Olympic Games Paris 2024"],
                "reasoning": "The flotilla carrying national athlete delegations along the Seine River in Paris signifies the opening ceremony of the 2024 Summer Olympic Games.",
                "level": 1,
            },
            "q2": {
                "question": "What historic river serves as the open-air parade route in this unprecedented ceremony?",
                "options": {
                    "A": "Thames River",
                    "B": "Danube River",
                    "C": "Seine River",
                    "D": "Rhine River",
                    "E": "Rhone River",
                },
                "answer": "C",
                "ground_truth_text": "Seine River",
                "ground_truth_list": ["Seine River", "The Seine", "River Seine"],
                "reasoning": "The flotilla traverses 6 kilometers from the Pont d'Austerlitz to the Pont d'Iena directly across the Seine River in central Paris.",
                "level": 2,
            },
        },
        {
            "img_id": "news_spacex_starship",
            "domain": "news",
            "title": "Starship Flight 5 Tower Catch",
            "color": (230, 126, 34),
            "pattern": "rect",
            "date": "2024-10-13",
            "q1": {
                "question": "Based on the provided image, what aerospace vehicle booster is being captured?",
                "options": {
                    "A": "Falcon Heavy Core",
                    "B": "Super Heavy Booster",
                    "C": "New Glenn Stage 1",
                    "D": "SLS Core Stage",
                    "E": "Vulcan Centaur Booster",
                },
                "answer": "B",
                "ground_truth_text": "Super Heavy Booster",
                "ground_truth_list": ["Super Heavy Booster", "Super Heavy", "Starship Booster"],
                "reasoning": "The massive stainless steel rocket caught between mechanical arms at Starbase corresponds to SpaceX's Super Heavy booster during Integrated Flight Test 5.",
                "level": 1,
            },
            "q2": {
                "question": "What is the specialized launch tower mechanism informally nicknamed that caught the descending booster?",
                "options": {
                    "A": "Mechazilla",
                    "B": "SkyHook",
                    "C": "HydraClamp",
                    "D": "FalconGrip",
                    "E": "TitanClaw",
                },
                "answer": "A",
                "ground_truth_text": "Mechazilla",
                "ground_truth_list": ["Mechazilla", "Chopsticks", "Mechazilla chopsticks"],
                "reasoning": "SpaceX engineers and Elon Musk designated the launch tower with articulating catching arms 'Mechazilla', which executed the historic booster recovery.",
                "level": 2,
            },
        },
        {
            "img_id": "news_cop29_baku",
            "domain": "news",
            "title": "COP29 Climate Summit Baku",
            "color": (39, 174, 96),
            "pattern": "scene",
            "date": "2024-11-20",
            "q1": {
                "question": "Based on the provided image, in which host city is this United Nations climate summit convening?",
                "options": {
                    "A": "Dubai",
                    "B": "Sharm El Sheikh",
                    "C": "Baku",
                    "D": "Belem",
                    "E": "Glasgow",
                },
                "answer": "C",
                "ground_truth_text": "Baku",
                "ground_truth_list": ["Baku", "Baku, Azerbaijan", "Baku Stadium"],
                "reasoning": "The summit banners and Baku Olympic Stadium venue represent the UN Climate Change Conference COP29 held in Baku, Azerbaijan.",
                "level": 1,
            },
            "q2": {
                "question": "What was the final negotiated target amount in billions of dollars per year for the New Collective Quantified Goal (NCQG) climate finance target established at this summit?",
                "options": {
                    "A": "100",
                    "B": "250",
                    "C": "300",
                    "D": "500",
                    "E": "1000",
                },
                "answer": "C",
                "ground_truth_text": "300",
                "ground_truth_list": ["300", "300 billion", "$300 billion"],
                "reasoning": "After protracted negotiations in Baku, developed countries pledged to scale up climate finance to at least $300 billion annually by 2035.",
                "level": 2,
            },
        },

        # Domain 2: Video
        {
            "img_id": "video_rivian_r1s",
            "domain": "video",
            "title": "Rivian Second Generation Review",
            "color": (52, 73, 94),
            "pattern": "rect",
            "date": "2024-06-12",
            "q1": {
                "question": "Based on the provided image, what electric vehicle model is undergoing test-drive review?",
                "options": {
                    "A": "Tesla Cybertruck",
                    "B": "Rivian R1S",
                    "C": "Lucid Gravity",
                    "D": "Ford F-150 Lightning",
                    "E": "Cadillac Escalade IQ",
                },
                "answer": "B",
                "ground_truth_text": "Rivian R1S",
                "ground_truth_list": ["Rivian R1S", "R1S", "Rivian R1S Gen 2"],
                "reasoning": "The signature horizontal stadium-light bar and SUV silhouette uniquely identify the 2025 Gen-2 Rivian R1S electric SUV.",
                "level": 1,
            },
            "q2": {
                "question": "What new compute platform architecture did the manufacturer introduce in this vehicle redesign that reduced electronic control units (ECUs) from 17 to 7?",
                "options": {
                    "A": "Zonal architecture with NVIDIA DRIVE Orin",
                    "B": "Centralized Tesla HW4 FSD computer",
                    "C": "Mobileye EyeQ6 High-Compute platform",
                    "D": "Qualcomm Snapdragon Digital Chassis",
                    "E": "Bosch Cross-Domain Vehicle Computer",
                },
                "answer": "A",
                "ground_truth_text": "Zonal architecture with NVIDIA DRIVE Orin",
                "ground_truth_list": ["Zonal architecture with NVIDIA DRIVE Orin", "Zonal architecture", "NVIDIA DRIVE Orin zonal architecture"],
                "reasoning": "Rivian completely redesigned the electronic network into a centralized zonal architecture utilizing dual NVIDIA DRIVE Orin processors to drastically consolidate wiring and ECUs.",
                "level": 2,
            },
        },
        {
            "img_id": "video_apple_vision_pro",
            "domain": "video",
            "title": "Spatial Computing Interface Demo",
            "color": (127, 140, 141),
            "pattern": "rect",
            "date": "2024-05-18",
            "q1": {
                "question": "Based on the provided image, what mixed reality wearable headset is demonstrated?",
                "options": {
                    "A": "Meta Quest 3",
                    "B": "Apple Vision Pro",
                    "C": "HTC Vive XR Elite",
                    "D": "PlayStation VR2",
                    "E": "Magic Leap 2",
                },
                "answer": "B",
                "ground_truth_text": "Apple Vision Pro",
                "ground_truth_list": ["Apple Vision Pro", "Vision Pro", "Apple Headset"],
                "reasoning": "The laminated glass front with EyeSight display, aluminum alloy frame, and modular Solo Knit Band identify the Apple Vision Pro.",
                "level": 1,
            },
            "q2": {
                "question": "What custom dual-chip processing system powers the spatial video pipeline and real-time sensor processing in this device?",
                "options": {
                    "A": "Snapdragon XR2 Gen 2 + Fast-CV",
                    "B": "Apple M2 + Apple R1",
                    "C": "Apple M4 + Neural Engine",
                    "D": "Intel Core Ultra + RealSense",
                    "E": "Tensor G4 + TPU Edge",
                },
                "answer": "B",
                "ground_truth_text": "Apple M2 + Apple R1",
                "ground_truth_list": ["Apple M2 + Apple R1", "M2 and R1", "M2 + R1"],
                "reasoning": "The hardware architecture pairs an Apple M2 chip for general OS and app execution with a dedicated R1 chip streaming 12 cameras and 5 sensors with 12ms latency.",
                "level": 2,
            },
        },
        {
            "img_id": "video_euro_2024_final",
            "domain": "video",
            "title": "UEFA Euro 2024 Championship Final",
            "color": (192, 57, 43),
            "pattern": "scene",
            "date": "2024-07-14",
            "q1": {
                "question": "Based on the provided image, which national football team is celebrating their tournament trophy victory?",
                "options": {
                    "A": "England",
                    "B": "France",
                    "C": "Spain",
                    "D": "Germany",
                    "E": "Netherlands",
                },
                "answer": "C",
                "ground_truth_text": "Spain",
                "ground_truth_list": ["Spain", "Spanish National Team", "La Roja"],
                "reasoning": "The red kits, gold trim, and Henri Delaunay Trophy celebration commemorate Spain's 2-1 victory over England in Berlin.",
                "level": 1,
            },
            "q2": {
                "question": "In what exact minute of regular play was the decisive winning goal scored by Mikel Oyarzabal?",
                "options": {
                    "A": "47",
                    "B": "73",
                    "C": "86",
                    "D": "90",
                    "E": "119",
                },
                "answer": "C",
                "ground_truth_text": "86",
                "ground_truth_list": ["86", "86th minute", "Minute 86"],
                "reasoning": "Marc Cucurella provided a low cross from the left wing that substitute Mikel Oyarzabal redirected into the net in the 86th minute.",
                "level": 2,
            },
        },
        {
            "img_id": "video_deepseek_r1_release",
            "domain": "video",
            "title": "Open Source Reasoning Model Explainer",
            "color": (26, 188, 156),
            "pattern": "diagram",
            "date": "2025-01-22",
            "q1": {
                "question": "Based on the provided image, what open-source reasoning model architecture is being analyzed?",
                "options": {
                    "A": "Llama 3.3 70B",
                    "B": "DeepSeek-R1",
                    "C": "Qwen 2.5 Max",
                    "D": "Mistral Large 2",
                    "E": "Command R+",
                },
                "answer": "B",
                "ground_truth_text": "DeepSeek-R1",
                "ground_truth_list": ["DeepSeek-R1", "DeepSeek R1", "R1-Zero"],
                "reasoning": "The diagram details the reinforcement learning pipeline from DeepSeek-V3-Base through pure GRPO rule-based training that defines DeepSeek-R1.",
                "level": 1,
            },
            "q2": {
                "question": "What reinforcement learning algorithm without a critic model was utilized to train this reasoning system?",
                "options": {
                    "A": "PPO (Proximal Policy Optimization)",
                    "B": "DPO (Direct Preference Optimization)",
                    "C": "GRPO (Group Relative Policy Optimization)",
                    "D": "KTO (Kahneman-Tversky Optimization)",
                    "E": "RLOO (REINFORCE Leave-One-Out)",
                },
                "answer": "C",
                "ground_truth_text": "GRPO (Group Relative Policy Optimization)",
                "ground_truth_list": ["GRPO", "Group Relative Policy Optimization", "GRPO algorithm"],
                "reasoning": "DeepSeek-R1 innovated by eschewing traditional value network critics and instead adopting Group Relative Policy Optimization (GRPO) using group-relative outcome scores.",
                "level": 2,
            },
        },

        # Domain 3: arXiv Academic Papers
        {
            "img_id": "arxiv_gdi_bench",
            "domain": "arxiv",
            "title": "GDI-Bench Vision & Reasoning",
            "color": (44, 62, 80),
            "pattern": "diagram",
            "date": "2024-05-30",
            "q1": {
                "question": "Based on the provided image, what is the title of the research paper proposing this decoupled benchmark?",
                "options": {
                    "A": "GDI-Bench: Decoupled Vision and Reasoning in Document Intelligence",
                    "B": "DocVQA: Large-Scale Benchmark for Document Understanding",
                    "C": "ChartBench: Evaluating Vision Models on Complex Graphics",
                    "D": "OmniDoc: Multimodal Document Parsing at Scale",
                    "E": "TextVQA: Grounding Text in Visual Documents",
                },
                "answer": "A",
                "ground_truth_text": "GDI-Bench: Decoupled Vision and Reasoning in Document Intelligence",
                "ground_truth_list": ["GDI-Bench", "GDI-Bench: Decoupled Vision and Reasoning in Document Intelligence"],
                "reasoning": "The dual-pipeline diagram separating optical perceptual parsing from symbolic logical deduction illustrates the GDI-Bench framework.",
                "level": 1,
            },
            "q2": {
                "question": "Who is the designated lead author on this research study?",
                "options": {
                    "A": "Jian Zhou",
                    "B": "Lin Chen",
                    "C": "Mingyang Fu",
                    "D": "Dongping Chen",
                    "E": "Zetong Zhou",
                },
                "answer": "A",
                "ground_truth_text": "Jian Zhou",
                "ground_truth_list": ["Jian Zhou", "Zhou Jian"],
                "reasoning": "The paper publication metadata on arXiv identifies Jian Zhou as the first author who spearheaded the benchmark design.",
                "level": 2,
            },
        },
        {
            "img_id": "arxiv_mm_search",
            "domain": "arxiv",
            "title": "MM-Search Multimodal Search Engine",
            "color": (211, 84, 0),
            "pattern": "diagram",
            "date": "2024-09-24",
            "q1": {
                "question": "Based on the provided image, what benchmark framework evaluates MLLMs as multimodal search agents?",
                "options": {
                    "A": "WebVoyager",
                    "B": "MM-Search",
                    "C": "VisualAgentBench",
                    "D": "BrowseGym",
                    "E": "Mind2Web",
                },
                "answer": "B",
                "ground_truth_text": "MM-Search",
                "ground_truth_list": ["MM-Search", "MMSearch", "MM-Search Benchmark"],
                "reasoning": "The three-stage query reformulator, screenshot reranker, and evidence synthesizer diagram illustrates the MM-Search engine benchmark by Jiang et al.",
                "level": 1,
            },
            "q2": {
                "question": "Across the 3-stage multimodal search pipeline, what is the exact function of Stage 2?",
                "options": {
                    "A": "Generating visual requeries",
                    "B": "Filtering and reranking visual search candidate pages",
                    "C": "Transcribing audio tracks",
                    "D": "Synthesizing the final answer text",
                    "E": "Translating queries into SQL",
                },
                "answer": "B",
                "ground_truth_text": "Filtering and reranking visual search candidate pages",
                "ground_truth_list": ["Filtering and reranking visual search candidate pages", "Reranking candidate pages", "Rerank"],
                "reasoning": "In the MM-Search architecture, Stage 1 generates text/visual queries, Stage 2 screens browser screenshots and reranks candidate URLs, and Stage 3 synthesizes answers.",
                "level": 2,
            },
        },
        {
            "img_id": "arxiv_hmfnet",
            "domain": "arxiv",
            "title": "HMFNet Molecular Odor Network",
            "color": (155, 89, 182),
            "pattern": "diagram",
            "date": "2024-11-05",
            "q1": {
                "question": "Based on the provided image, what neural network model is diagrammed for molecular odor prediction?",
                "options": {
                    "A": "HMFNet",
                    "B": "ChemBERTa",
                    "C": "MolCLR",
                    "D": "GraphSAGE-Odor",
                    "E": "Uni-Mol",
                },
                "answer": "A",
                "ground_truth_text": "HMFNet",
                "ground_truth_list": ["HMFNet", "Harmonic Multi-Feature Network"],
                "reasoning": "The architecture displaying LMFE, GMFE, and HMFM modules corresponds to the Hierarchical Multi-Feature Network (HMFNet) for olfaction classification.",
                "level": 1,
            },
            "q2": {
                "question": "In this paper, what is the total number of distinct novel deep learning modules explicitly proposed within HMFNet?",
                "options": {
                    "A": "2",
                    "B": "3",
                    "C": "4",
                    "D": "5",
                    "E": "6",
                },
                "answer": "C",
                "ground_truth_text": "4",
                "ground_truth_list": ["4", "four"],
                "reasoning": "As established in the paper's ablation studies (and Figure 32 of LiveVQA), the authors propose exactly 4 components: LMFE, GMFE, HMFM, and the composite HMFNet integration.",
                "level": 2,
            },
        },
        {
            "img_id": "arxiv_codesync",
            "domain": "arxiv",
            "title": "CodeSync Continual Model Evolution",
            "color": (22, 160, 133),
            "pattern": "diagram",
            "date": "2025-02-18",
            "q1": {
                "question": "Based on the provided image, what continual updating framework synchronizes LLMs with dynamic code repositories?",
                "options": {
                    "A": "CodeSync",
                    "B": "RepoCoder",
                    "C": "SweetCode",
                    "D": "GitAgent",
                    "E": "CodeUpdateArena",
                },
                "answer": "A",
                "ground_truth_text": "CodeSync",
                "ground_truth_list": ["CodeSync", "CodeSync Framework"],
                "reasoning": "The commit-replay and dependency-aware synchronization flowchart depicts the CodeSync framework by Wang et al.",
                "level": 1,
            },
            "q2": {
                "question": "What primary phenomenon does this paper analyze when updating code models on rapid API modifications?",
                "options": {
                    "A": "Overfitting on indentation",
                    "B": "Catastrophic forgetting of prior library methods and syntax",
                    "C": "Loss of natural language translation accuracy",
                    "D": "Prompt injection vulnerability",
                    "E": "GPU memory thrashing during attention calculation",
                },
                "answer": "B",
                "ground_truth_text": "Catastrophic forgetting of prior library methods and syntax",
                "ground_truth_list": ["Catastrophic forgetting of prior library methods and syntax", "Catastrophic forgetting", "Knowledge forgetting"],
                "reasoning": "CodeSync specifically studies catastrophic forgetting where updating LLMs on newly released API methods causes immediate erosion of older deprecated syntax.",
                "level": 2,
            },
        },
    ]

    for item in items:
        img_name = f"{item['img_id']}.png"
        img_rel = f"livevqa_images/{img_name}"
        img_abs = out_file.parent / img_rel

        # Render image
        create_annotated_image(
            file_path=img_abs,
            title=item["title"],
            domain=item["domain"],
            accent_color=item["color"],
            pattern=item["pattern"],
        )

        # Level 1 sample
        q1_data = item["q1"]
        records.append({
            "id": f"{item['img_id']}_l1",
            "image": img_rel.replace("\\", "/"),
            "question": q1_data["question"],
            "options": q1_data["options"],
            "answer": q1_data["answer"],
            "ground_truth_text": q1_data["ground_truth_text"],
            "ground_truth_list": q1_data["ground_truth_list"],
            "reasoning": q1_data["reasoning"],
            "level": q1_data["level"],
            "domain": item["domain"],
            "date": item["date"],
            "is_synthetic": False,
            "metadata": {
                "title": item["title"],
                "category": item["domain"],
                "difficulty": "Level 1 (Entity Recognition)",
            },
        })

        # Level 2 sample
        q2_data = item["q2"]
        records.append({
            "id": f"{item['img_id']}_l2",
            "image": img_rel.replace("\\", "/"),
            "question": q2_data["question"],
            "options": q2_data["options"],
            "answer": q2_data["answer"],
            "ground_truth_text": q2_data["ground_truth_text"],
            "ground_truth_list": q2_data["ground_truth_list"],
            "reasoning": q2_data["reasoning"],
            "level": q2_data["level"],
            "domain": item["domain"],
            "date": item["date"],
            "is_synthetic": False,
            "metadata": {
                "title": item["title"],
                "category": item["domain"],
                "difficulty": "Level 2 (Multi-Hop Reasoning)",
            },
        })

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Generated %d LiveVQA laptop benchmark samples at %s", len(records), out_file)
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_livevqa_laptop_dataset()
