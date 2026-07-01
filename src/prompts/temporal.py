"""Temporal grounding prompts."""

def build_temporal_prompt(
    model_family: str,
    query_text: str,
    text_only: bool = False,
    selected_frame_ids: list[int] | None = None,
) -> str:
    """Build a temporal grounding prompt specification."""
    query = query_text.strip()

    if model_family == "gemini":
        return (
            "Answer with time ranges and do not output explanation. "
            f"What is the single time range corresponding to the text query: \"{query}\"? "
            "Output format:"
            "[start, end]"
        )

    if model_family == "gpt":
        return (
            "The input images are frames from a video. Output the frame indexes that "
            f"correspond to the text query: \"{query}\". Only output the index range, "
            "for example, 2-4, 6-8."
        )

    if model_family == "internvl":
        return (
            f"Give you a textual query: {query}\n"
            "When does the described content occur in the video?\n"
            "Please return the timestamp in seconds. "
            "Output format:"
            "[start, end]"
        )

    if model_family == "qwen":
        return (
            f"Give you a textual query: {query}\n"
            "When does the described content occur in the video?\n"
            "Please return the timestamp in seconds. "
            "Output format:"
            "[start, end]"
        )
    # if domain == "animal":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The video contains footage of animal behavior, and your task is to identify the exact time interval [start, end] in seconds when the described action occurs.  \n"
        #     "\n"
        #     "**Key instructions**:  \n"
        #     "1. **Zero-shot temporal grounding**: Output exactly one [start, end] span in seconds, **not assuming the action starts at 0.0**. Prioritize aligning with the **ground truth temporal bounds** (e.g., [6.0, 16.5] for \"rhinoceros walking\") and avoid overlapping or incorrect intervals.  \n"
        #     "2. **Domain specificity**: Focus on animal behaviors (e.g., grooming, movement, stillness, feeding, vocalization). Use **behavior-specific cues** (e.g., \"sensing surroundings\" implies vigilance behavior, \"sharing food\" implies social interaction) to infer precise temporal alignment.  \n"
        #     "3. **Precision**: Match the action’s **exact temporal span** by analyzing the entire video’s timeline. Avoid generic spans like [0.0, X] or [X, Y] that do not reflect the action’s actual duration. For example, \"singing nightingale\" should map to vocalization peaks, not generic movement intervals.  \n"
        #     "4. **Output format**: Return only the final answer in the format [start, end], with numerical values in seconds. Do not include explanations or additional text."
        # )

    # if domain == "industry":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The task is to identify the exact time interval in seconds when the described action occurs in the video. Output **only one** [start, end] span, where \"start\" is the first frame the action begins and \"end\" is the last frame the action concludes.  \n"
        #     "**Key Instructions:**  \n"
        #     "1. **Precision:** Align the span with the ground truth temporal bounds (e.g., [39.790154, 40.190154] for \"socket 3 being touched\"). Avoid broad guesses like [0.0, 61.0].  \n"
        #     "2. **Single Span:** Do not output multiple intervals or overlapping ranges.  \n"
        #     "3. **Zero-Shot Generalization:** If the query refers to an object (e.g., \"power supply\") or action (e.g., \"put screwdriver\"), infer the exact moment the described event occurs, even if the video does not explicitly label it.  \n"
        #     "4. **Formatting:** Use **seconds** (not frames) and ensure numerical precision (e.g., [142.973117, 153.440117] for \"power supply being touched\").  \n"
        #     "5. **Edge Cases:** For ambiguous queries (e.g., \"take bolt\"), prioritize the most temporally specific segment matching the query, even if the ground truth is narrow (e.g., [5.155, 5.905])."
        # )

    # if domain == "surgery":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The query describes a medical tool (e.g., \"scalpel,\" \"tweezers,\" \"gauze\") and its action (e.g., \"incision,\" \"dissection,\" \"hemostasis\") in a surgical procedure. Your task is to identify the exact time interval [start, end] in seconds when this tool-action pair occurs in the video.  \n"
        #     "\n"
        #     "Key guidelines:  \n"
        #     "1. **Stage Mapping (Precise Ranges)**:  \n"
        #     "   - **Early Stage** (e.g., \"incision,\" \"skin preparation,\" \"design\"): Use [0.0, 600.0] as a guideline but infer exact timing based on procedural flow. For example:  \n"
        #     "     - \"Incision\" typically occurs in [15.0, 30.0] (sharp start), [45.0, 60.0] (delayed start).  \n"
        #     "     - \"Design\" or \"skin preparation\" falls in [0.0, 600.0], but avoid zero-padding unless explicitly implied (e.g., \"preparation begins at 0.0\").  \n"
        #     "   - **Mid-Stage** (e.g., \"dissection,\" \"clipping,\" \"suturing\"): Occurs in [600.0, 1200.0]. For actions like \"dissection,\" use [600.0, 900.0] as a baseline.  \n"
        #     "   - **Late Stage** (e.g., \"hemostasis,\" \"closure,\" \"removal\"): Occurs in [1200.0, 1800.0], but adjust for prolonged procedures (e.g., \"closure\" may overlap mid-late stages: [900.0, 1500.0]).  \n"
        #     "   - **Action-Specific Adjustments**:  \n"
        #     "     - \"Hemostasis\" is late-stage: [1200.0, 1800.0].  \n"
        #     "     - \"Closure\" may span mid-late: [900.0, 1500.0].  \n"
        #     "     - \"Anesthesia\" is early-stage: [0.0, 600.0], but avoid zero-padding unless explicitly stated.  \n"
        #     "2. **Tool Inference (Explicit Mapping)**:  \n"
        #     "   - If the query lacks a tool name, infer based on action:  \n"
        #     "     - \"Incision\" → \"scalpel\" (e.g., [15.0, 30.0]).  \n"
        #     "     - \"Dissection\" → \"forceps\" or \"scissors\" (e.g., [600.0, 900.0]).  \n"
        #     "     - \"Hemostasis\" → \"sutures\" or \"clips\" (e.g., [1200.0, 1800.0]).  \n"
        #     "     - \"Design\" → \"scalpel\" or \"surgical marker\" (early-stage).  \n"
        #     "3. **Temporal Precision**:  \n"
        #     "   - Output a single [start, end] span with timestamps rounded to one decimal place (e.g., [30.2, 52.2]).  \n"
        #     "   - Align with procedural flow: Avoid zero-padding (e.g., [0.0, X]) unless explicitly implied by the query (e.g., \"preparation begins at 0.0\").  \n"
        #     "4. **Edge Case Handling**:  \n"
        #     "   - For ambiguous phrases like \"Other hands left during design,\" prioritize early-stage ranges ([0.0, 600.0]) but avoid zero-padding.  \n"
        #     "   - For actions like \"closure\" or \"removal,\" consider mid-late stages ([900.0, 1500.0]) if the procedure is prolonged.  \n"
        #     "5. **Format**: Return only the timestamp interval in seconds, without additional text."
        # )

    # if domain == "safety":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "When does the described content occur in the video?  \n"
        #     "**Key Instructions:**  \n"
        #     "1. Output **exactly one** time interval in seconds, formatted as `[start, end]` (one decimal place).  \n"
        #     "2. **Prioritize precise temporal alignment**: Focus on the **exact moment** the event begins and ends, avoiding overgeneralization (e.g., do not include pre- or post-event footage). Anchor the span to the **first visible action** of the described event and its **final state** (e.g., collision completion, object disappearance).  \n"
        #     "3. **Use domain-specific terms** to narrow the span:  \n"
        #     "   - *Camera vehicle* (e.g., \"the camera vehicle turns left\" → focus on its motion, not surrounding context).  \n"
        #     "   - *Anomalous vehicle* (e.g., \"a vehicle other than the camera vehicle collides\" → isolate its actions, not the camera’s).  \n"
        #     "   - *Roadway direction* (left/right, e.g., \"moves from right to left\" → align with directional motion).  \n"
        #     "   - *Action sequences* (e.g., \"hit [object] then ran away\" → span must cover **entire sequence**, not individual steps).  \n"
        #     "4. **Match the exact event**: If the query specifies \"a vehicle *other than the camera vehicle* goes out of control,\" ensure the span excludes the camera vehicle’s actions (e.g., focus on anomalous vehicle’s motion, not camera’s lateral movement).  \n"
        #     "5. **Handle sequential actions**: For multi-step events (e.g., \"stole a car, then drove away\"), capture the **entire temporal span** of the sequence, not isolated steps. Validate against ground truth to ensure alignment with the **collision’s start and end** (e.g., [4.7, 6.0] for a collision, not [0.0, 9.5]).  \n"
        #     "6. **Avoid ambiguity**: If the query includes spatial/temporal details (e.g., \"a person wearing striped clothes on the right side of the mall road\"), use these to **anchor the span** to the specific instance (e.g., align with the moment the person is visible on the right side).  \n"
        #     "7. **Validate against ground truth**: For known events (e.g., \"collision\"), ensure the span aligns with the **collision’s start and end**, not surrounding context (e.g., [2.1, 7.1] for a collision, not [0.0, 3.1]).  \n"
        #     "8. **Avoid default start times**: Do not assume events begin at 0.0. Instead, identify the **first visible action** of the described event (e.g., [45.7, 57.2] for a boy wiping a table, not [0.0, 4.6]).  \n"
        #     "9. **Prioritize specificity**: If the query includes directional or spatial details (e.g., \"camera vehicle moves laterally in the same direction\"), ensure the span reflects the **exact motion** (e.g., [2.5, 6.6] for lateral movement + collision, not [0.0, 7.8])."
        # )

    # if domain == "sports":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "When does the described content occur in the video?  \n"
        #     "Please return the timestamp in seconds. Output format: [start, end]  \n"
        #     "\n"
        #     "**Instructions for Zero-Shot Temporal Grounding:**  \n"
        #     "1. **Prioritize precise, narrow spans**: Identify the exact time interval for the event, avoiding overly broad ranges (e.g., [0.0, 6.5] for a short action). Use domain-specific knowledge to match the event’s known duration. For example:  \n"
        #     "   - **Volleyball**: \"Serve\" occurs at the start of a rally (e.g., [0.52, 2.6]), \"spike\" is a rapid motion (e.g., [5.24, 5.96]), and \"dink\" is a short, precise action (e.g., [3.56, 4.12]).  \n"
        #     "   - **Basketball**: \"Drive\" spans a play (e.g., [9.04, 10.52]), while \"dribble\" occurs during continuous movement (e.g., [0.04, 6.2]).  \n"
        #     "   - **Football**: \"Field goal attempt\" happens near the end of a possession (e.g., [29.7, 30.1]), and \"long pass\" occurs during offensive plays (e.g., [49.92, 50.24]).  \n"
        #     "   - **Soccer**: \"Kicker kicks a placed ball\" is a brief action (e.g., [20.702, 21.435]).  \n"
        #     "2. **Avoid generic assumptions**: Do not default to pre-existing examples (e.g., using [5.24, 5.96] for \"serve\" instead of the correct [0.52, 2.6]). Instead, infer based on the event’s nature and context.  \n"
        #     "3. **Single span output**: Return exactly one [start, end] interval. Ensure the span aligns with the event’s duration (e.g., a \"defend\" in volleyball lasts ~0.5–1.5 seconds).  \n"
        #     "4. **Use context for estimation**: If uncertain, apply domain knowledge. For example, \"first pass\" in volleyball typically occurs early in the rally (e.g., [2.48, 3.12]), while \"drive\" in basketball spans a longer play.  \n"
        #     "5. **Prioritize accuracy over coverage**: If the event is brief (e.g., \"dink\"), avoid overextending the span. For multi-moment events (e.g., a \"play\" in football), focus on the critical action (e.g., the kick, not the entire play).  \n"
        #     "6. **Explicitly reject broad spans**: Unless the event spans the entire video (e.g., \"game introduction\"), avoid ranges like [0.0, 6.7]."
        # )


        # return (
        #     f"""Give you a textual query: {query}  
        #     When does the described content occur in the video?  
        #     Please return the timestamp in seconds. Output format: [start, end]  

        #     **Instructions for Zero-Shot Temporal Grounding**  
        #     1. **Domain-Specific Timing Knowledge**:  
        #     - **Football**:  
        #         - "Kickoff" occurs at the start of the video (e.g., [0.0, 4.3]).  
        #         - "Clearance" or "steal" happens mid-play (e.g., [9.16, 9.44]).  
        #         - "Tackle" occurs during active gameplay (e.g., [20.32, 20.76]).  
        #     - **Volleyball**:  
        #         - "First pass" occurs early in the rally (e.g., [3.4, 3.64]).  
        #         - "Protect" refers to defensive actions during play (e.g., [5.6, 6.64]).  
        #         - "Adjust" occurs mid-rally (e.g., [2.92, 3.4]).  
        #     - **Basketball**:  
        #         - "Pass-inbound" occurs at the start of the video (e.g., [0.0, 2.54]).  
        #         - "Screen" happens mid-play (e.g., [10.12, 10.84]).  

        #     2. **Precision Requirements**:  
        #     - Output **exactly one** [start, end] interval.  
        #     - Align with ground truth spans using temporal overlap (IoU).  
        #     - Ensure timestamps fall **within the video’s visible duration** (e.g., if the video ends at 30 seconds, avoid extrapolating beyond 30).  

        #     3. **Generalizable Strategy**:  
        #     - **Prioritize explicit cues**: If the query contains a term with domain-specific timing (e.g., "first pass" = early rally), use the corresponding interval.  
        #     - **Default to mid-video** for ambiguous queries (e.g., [video_duration/2 - 2, video_duration/2 + 2]).  
        #     - **Avoid extrapolation**: Use the video’s actual duration (e.g., if the video is 60 seconds long, mid-video is [28.68, 31.32]).  
        #     - **Sport inference**: If the query mentions a sport (e.g., "volleyball adjust"), apply the relevant domain rules. If no sport is specified, default to mid-video timing.  

        #     4. **Edge Case Handling**:  
        #     - For terms not in domain-specific knowledge (e.g., "interference shot" in basketball), use **generalizable strategies** and infer timing based on sport-specific context.  
        #     - If no temporal cues are present, output the **mid-video interval** calculated using the video’s actual duration.  
        #     - **Refine mid-video defaults**: If the query includes a sport, calculate mid-video as [video_duration/2 - 2, video_duration/2 + 2]. If no sport is specified, default to [18.68, 20.38] (as in examples).  

        #     5. **Critical Adjustments**:  
        #     - **Avoid overfitting to example timestamps**: Use domain rules as guidelines, not fixed intervals. For example, "first pass" in volleyball is early, but the exact timing depends on the rally’s progression.  
        #     - **Use sport-specific context**: For ambiguous terms like "2-point shot" in basketball, infer that it occurs during active gameplay (not at the start or mid-play), and estimate timing based on typical shot durations (e.g., [25.0, 27.0]).  
        #     - **Explicitly handle mismatches**: If the query contains a term not in domain knowledge (e.g., "interference shot"), use the sport’s general timing (e.g., basketball shots occur during active gameplay) and estimate mid-play intervals."""
        # )


    # if domain == "animal":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "Identify the exact time interval [start, end] in seconds when the described action or object behavior occurs in the video. **Prioritize specificity**: Focus strictly on the precise event or behavior mentioned (e.g., \"repeatedly scratching itself with hind paw,\" \"sensing environment,\" \"being attacked\") and avoid generic interpretations. **Ensure full duration coverage**: The span must fully encompass the event's timeline, starting at the first observable instance and ending at the last moment the behavior is visible. **Avoid default start times**: Do not assume events occur at 0.0; infer the actual onset from the video. **Precision requirement**: Output **exactly one** [start, end] span, avoiding overlapping or ambiguous intervals. **Generalizability**: Do not rely on prior knowledge of species, scenes, or video content—base your answer solely on the query and the video's temporal structure.  \n"
        #     "```  \n"
        #     "\n"
        #     "### Rationale for Key Enhancements:  \n"
        #     "1. **Explicit Specificity**: The examples show the model struggles with generic phrasing (e.g., \"the monkey is eating\") and defaults to incorrect spans. Emphasizing precise behaviors (e.g., \"repeatedly scratching\") ensures alignment with ground truth.  \n"
        #     "2. **Full Duration Coverage**: The feedback highlights truncated spans (e.g., [0.0, 2.7] vs. [16.6, 21.9]). The revised instruction mandates starting at the first observable instance and ending at the last, avoiding partial ranges.  \n"
        #     "3. **Avoid Default Start Times**: The model previously assumed events begin at 0.0, leading to poor IoU. The instruction explicitly prohibits this, forcing the model to infer temporal boundaries from the video.  \n"
        #     "4. **Single Span Precision**: The feedback stresses the need for one exact interval. This avoids ambiguity and ensures alignment with ground truth.  \n"
        #     "5. **No Prior Knowledge Dependency**: The task requires zero-shot generalization, so the model must avoid assumptions about species or scenes and focus solely on the query and video content."
        # )

    # if domain == "industry":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "When does the described content occur in the video?  \n"
        #     "Please return **exactly one** [start, end] span in seconds, where:  \n"
        #     "1. **start** is the earliest timestamp (in seconds) the content appears  \n"
        #     "2. **end** is the latest timestamp (in seconds) the content persists  \n"
        #     "3. If the event spans multiple segments, output the **tightest possible span** covering all instances (i.e., the minimal [max_start, min_end] that fully contains all occurrences)  \n"
        #     "4. If the event is ambiguous (e.g., no clear temporal boundaries) or spans the entire video, use [0.0, {video_duration}]  \n"
        #     "5. **Prioritize precision**: Avoid over-estimating (e.g., [0.0, 158.4] for a localized event) by explicitly locating the event’s temporal boundaries  \n"
        #     "6. **Format**: [start, end] (no extra text)  \n"
        #     "\n"
        #     "**Key Enhancements:**  \n"
        #     "- **Decompose queries**: Break down the query into object/action components (e.g., \"red angled perforated bar\" → \"bar\", \"red\", \"angled\", \"perforated\") and map them to the video’s content.  \n"
        #     "- **Attribute alignment**: Ensure color, shape, and material descriptors (e.g., \"white\", \"angled\", \"perforated\") are explicitly matched to the video’s visual elements.  \n"
        #     "- **Action specificity**: For verbs like \"take\" or \"put,\" align with precise motion boundaries (e.g., [4.063, 4.733] for \"put screwdriver\").  \n"
        #     "- **Zero-shot grounding**: If the query describes a novel object/action not explicitly present in the video, output [0.0, {video_duration}] only if no clear temporal boundaries exist.  \n"
        #     "- **Ground truth alignment**: Use precise temporal boundaries (e.g., [24.749, 35.316] for \"welder station being touched\") to maximize Temporal IoU.  \n"
        #     "- **Avoid overgeneralization**: Never assume the event spans the entire video unless explicitly ambiguous (e.g., \"background noise\" or \"general activity\")."
        # )

    # if domain == "sports":
        # return (
        #     f"Give you a textual query: {query}\n"
        #     "When does the described content occur in the video?\n"
        #     "Please return the timestamp in seconds. Output format:[start, end]"
        # )

    # if domain == "surgery":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The query describes a medical tool (e.g., tweezers, scalpel, gauze, mouth gag, suction tip) and its use during a specific surgical phase (e.g., incision, hemostasis, dissection, anesthesia) or an unspecified phase (e.g., preparation, irrigation).  \n"
        #     "When does the described content occur in the video?  \n"
        #     "**Instructions for the model**:  \n"
        #     "1. **Zero-shot temporal grounding**: Identify the exact time span [start, end] in seconds where the tool is actively used during the specified surgical phase or inferred phase.  \n"
        #     "2. **Phase mapping**:  \n"
        #     "   - *Anesthesia*: Tools like mouth gags, laryngoscopes, or intubation devices are used **early** ([0.0–20.0]) or **mid-phase** ([10.0–40.0]) if context suggests later use.  \n"
        #     "   - *Incision*: Tools like scalpels, tweezers, electrocautery, or needle holders are used **after initial setup** ([10–60 seconds]).  \n"
        #     "   - *Hemostasis*: Gauze, clamps, hemostats, or suture materials are used **after incision** ([20–70 seconds]).  \n"
        #     "   - *Dissection*: Forceps, scissors, suction tips, or electrocautery are used **after incision** ([30–90 seconds]).  \n"
        #     "   - *Unspecified phases*:  \n"
        #     "     - *Preparation* → [0.0–40.0] (initial setup).  \n"
        #     "     - *Irrigation* → [30–70] (mid-dissection).  \n"
        #     "     - *Clipping/cutting* → [30–90] (dissection).  \n"
        #     "     - *Other unspecified* → map to closest phase (e.g., \"setup\" → [0.0–40.0], \"irrigation\" → [30–70]).  \n"
        #     "3. **Precision**: Output **exactly one** [start, end] span. Avoid early timestamps unless phase explicitly allows (e.g., \"anesthesia\") or inferred phase is early (e.g., \"preparation\").  \n"
        #     "4. **Alignment**: Ensure span matches phase’s typical timing. For example:  \n"
        #     "   - \"Specimen bag during clipping and cutting\" → [30–90] (dissection).  \n"
        #     "   - \"Grasper during preparation\" → [0.0–40.0] (initial setup).  \n"
        #     "   - \"Suction tip during dissection\" → [30–70] (mid-phase).  \n"
        #     "5. **Generalizable strategy**:  \n"
        #     "   - If tool not listed for phase, infer role based on phase’s typical tools.  \n"
        #     "   - For ambiguous queries (e.g., \"Other hands right during irrigation\"), prioritize phase-specific ranges over tool defaults.  \n"
        #     "6. **Edge cases**:  \n"
        #     "   - Avoid spans conflicting with phase timelines (e.g., [0.0–5.2] for \"tweezers during incision\" if ground truth is [32.8–58.8]).  \n"
        #     "   - For tools like \"specimen bag\" or \"grasper\" not explicitly listed, map to closest phase (e.g., \"clipping and cutting\" → dissection).  \n"
        #     "7. **Validation**: Ensure span aligns with phase’s typical timing and avoids overgeneralization. For example:  \n"
        #     "   - \"Own hands left during anesthesia\" → [10.0–40.0] (mid-phase) if used later, not [0.0–40.0].  \n"
        #     "   - \"Syringe during anesthesia\" → [0.0–20.0] (early) or [10.0–40.0] (initial setup) based on context."
        # )

    # if domain == "safety":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The video describes events involving specific actions (e.g., collisions, camera movements like lateral shifts or road crossings, or object interactions like entering/leaving a space). When does the described content occur in the video?  \n"
        #     "Please return the **most precise** timestamp span in seconds, formatted as [start, end]. Focus on the exact temporal boundaries of the event (e.g., the moment the anomalous car appears, the collision occurs, or the subject enters/leaves the frame). Ensure the span is as tight as possible to maximize Temporal IoU.  \n"
        #     "```  \n"
        #     "\n"
        #     "### Key Improvements:  \n"
        #     "1. **Domain-Specific Cues**: Explicitly mentions camera movements (e.g., \"lateral shifts,\" \"turning into a road\") and object interactions (e.g., \"entering/leaving a space\") to guide the model toward precise temporal anchors.  \n"
        #     "2. **Precision Emphasis**: Prioritizes tight spans by focusing on \"exact temporal boundaries\" (e.g., collision moments, appearance/disappearance).  \n"
        #     "3. **Zero-Shot Guidance**: Clarifies the need for a single, accurate span to avoid over-approximation (e.g., [0.0, 9.5] → [2.9, 7.3]).  \n"
        #     "4. **Task Alignment**: Reinforces the goal of matching ground truth intervals via Temporal IoU, ensuring the model avoids broad or misaligned predictions."
        # )

    # if dataset_name == "american_football":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "Identify the exact time interval [start, end] in seconds where the described event occurs in the video.  \n"
        #     "**Key instructions:**  \n"
        #     "1. Output **exactly one** [start, end] span, ensuring precise alignment with the event's temporal boundaries.  \n"
        #     "2. Prioritize the **full duration** of the described action (e.g., \"runs with the ball\" includes the entire motion, not just the handoff).  \n"
        #     "3. Use **seconds as the unit** (e.g., 15.09 for 15 seconds and 9 hundredths).  \n"
        #     "4. For multi-part events (e.g., \"catches a pass *and* the play continues until stopped\"), capture the **entire span** from the first action to the final conclusion.  \n"
        #     "5. **Focus exclusively on explicit temporal cues** in the query (e.g., \"until stopped\" indicates an endpoint, \"after the whistle\" defines a start). Avoid assumptions about timing.  \n"
        #     "6. If uncertain, **anchor the span to the most specific temporal markers** in the query (e.g., \"at 24.6 seconds\" or \"until the ball is out of bounds\").  \n"
        #     "7. Format strictly as **[start, end]** without additional text."
        # )

    # # if dataset_name == "animal_kingdom":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The task is to determine the exact time interval [start, end] in seconds when the described action occurs in the video.  \n"
        #     "**Key instructions for the model**:  \n"
        #     "1. **Zero-shot temporal grounding**: Focus exclusively on the time span where the query's description matches the video's content. Ignore unrelated scenes, background activity, or partial actions.  \n"
        #     "2. **Single precise span**: Return **exactly one** [start, end] interval, formatted as a list (e.g., [15.4, 17.5]). Do not assume the action starts at 0.0, spans the entire video, or occurs at the video’s beginning/end.  \n"
        #     "3. **Full action duration**: Ensure the interval captures the **entire duration** of the described action. For example:  \n"
        #     "   - For \"keeping still\" or \"exploring,\" assume a **short, contiguous duration** (1–5 seconds).  \n"
        #     "   - For \"walking\" or \"eating,\" assume a **longer, continuous duration** (5–20 seconds).  \n"
        #     "4. **Search entire video**: The action may occur **anywhere** in the video (e.g., mid-video). Verify the span aligns with the **most temporally aligned** instance of the action if multiple occur.  \n"
        #     "5. **Strict temporal overlap**: The predicted span must **overlap meaningfully** with the ground truth. Use the following criteria:  \n"
        #     "   - The intersection of predicted and ground truth spans must exceed **50% of the ground truth duration**.  \n"
        #     "   - Avoid spans like [0.0, 1.0] for events happening at [15.4, 17.5].  \n"
        #     "6. **Units**: All timestamps must be in **seconds** (e.g., 49.0, 56.1). Do not use formats like \"49s\" or \"56:1\".  \n"
        #     "7. **Example alignment**:  \n"
        #     "   - For \"The lanius excubitor is eating,\" output a longer span (e.g., [15.4, 17.5]).  \n"
        #     "   - For \"The lion is keeping still,\" output a short, precise span (e.g., [5.2, 6.7]).  \n"
        #     "8. **Critical clarification**: If the video contains multiple instances of the described action, select the **most temporally aligned** span that fully matches the query’s description. Avoid spans that include unrelated content or partial actions.  \n"
        #     "9. **Avoid common pitfalls**:  \n"
        #     "   - Do not infer action start/end times from implicit cues (e.g., \"the animal is moving\" does not imply the action starts at 0.0).  \n"
        #     "   - Prioritize **temporal alignment** over assumption-based guesses. For example, if the ground truth is [13.7, 17.4], avoid [0.0, 3.8] and instead search for spans in the mid-video range.  \n"
        #     "10. **Domain-specific focus**: Actions like \"hugging\" or \"walking\" require **precise temporal alignment** with the video’s content, even if the action is brief or prolonged."
        # )

    # # if dataset_name == "cholectrack20":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The task is to determine the exact time interval (in seconds) during which the described action occurs in the video. Focus on **zero-shot temporal grounding**, meaning you must infer the time span **without prior knowledge of the video's content** or external context, and **without relying on example spans** provided in the prompt.  \n"
        #     "\n"
        #     "### Key Requirements:  \n"
        #     "1. **Output exactly one time span** in the format `[start, end]` (seconds), using **numerical precision** (e.g., `[13.76, 43.76]`). Avoid vague terms like \"early,\" \"late,\" or \"during\" in the output.  \n"
        #     "2. **Prioritize precision over generality**: If the query refers to a tool (e.g., \"scissors,\" \"specimen bag,\" \"bipolar\") and an action (e.g., \"clipping and cutting,\" \"gallbladder packaging\"), focus on the **exact moment the tool is actively performing the action**. Avoid generic timestamps like \"first 10 seconds\" or \"toward the end.\"  \n"
        #     "3. **Assume the action occurs once**: If the query describes a specific action (e.g., \"hook during gallbladder dissection\"), infer the **precise temporal window** when this action is performed. Do not assume the action repeats or spans multiple intervals.  \n"
        #     "4. **Avoid overfitting to examples**: Do not copy spans from previous examples (e.g., `[107.272, 140.272]` for \"clipper during clipping and cutting\"). Instead, infer the span from the query alone, even if no examples are provided.  \n"
        #     "5. **Use Temporal IoU as a guide**: If the ground truth span is `[S, E]`, your output `[s, e]` should maximize overlap with `[S, E]` to ensure high Temporal IoU. However, since the ground truth is unknown, focus on aligning the span with the **most likely temporal boundaries** of the action based on the query.  \n"
        #     "\n"
        #     "### Strategy for Zero-Shot Inference:  \n"
        #     "- **Identify the tool and action**: If the query includes a tool (e.g., \"scissors\") and an action (e.g., \"clipping and cutting\"), infer the time when the tool is **actively involved** in the action.  \n"
        #     "- **Estimate a narrow span**: If the query lacks explicit timing cues, output a **narrow span** centered around the most likely moment the action occurs. For example, for \"scissors during clipping and cutting,\" assume the action occurs in the middle of the video (e.g., `[50.0, 70.0]`).  \n"
        #     "- **Avoid generic ranges**: Do not use broad ranges like \"first 10 seconds\" or \"toward the end.\" Instead, use precise numerical values to reflect the action's likely duration.  \n"
        #     "\n"
        #     "### Domain-Specific Notes:  \n"
        #     "- In surgical videos, actions like \"clipping and cutting\" or \"gallbladder packaging\" typically occur in **specific, short intervals** (e.g., 10–30 seconds).  \n"
        #     "- Tools like \"scissors,\" \"specimen bag,\" or \"bipolar\" are often used for **precise, time-bound tasks**, so their active use aligns with **short, focused temporal windows**.  \n"
        #     "- If the query lacks explicit timing cues, assume the action occurs **mid-video** (e.g., around 50–100 seconds) and use a narrow span to reflect the action's brevity."
        # )

    # # if dataset_name == "dota":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "Identify the **exact time interval [start, end]** in seconds when the **specific event** described in the query occurs in the video. Focus **strictly on the temporal context of the event** (e.g., \"collision,\" \"anomaly,\" \"vehicle interaction\") and align with the ground truth span by:  \n"
        #     "1. **Isolate the primary event** (e.g., \"collision\") as the core focus, even if the query includes ancillary details (e.g., \"anomalous car visible\"). Exclude pre-event setup (e.g., vehicle movement) or post-event aftermath (e.g., debris scattering).  \n"
        #     "2. **Capture the full duration** of the event (e.g., [4.4, 7.2] for a collision) by anchoring to **exact start/end moments** (e.g., the instant of impact and the moment the collision concludes). Avoid partial spans or overly broad ranges.  \n"
        #     "3. **Output exactly one [start, end] span** without ambiguity, ensuring alignment with the **short, action-driven nature** of video events (e.g., [1.6, 5.7] for a collision, not [0.0, 10.2]).  \n"
        #     "\n"
        #     "**Key Enhancements:**  \n"
        #     "- **Event-Centric Focus**: Prioritize the **specific event** (e.g., \"collision\") over generic video segments or ancillary details (e.g., \"anomalous car visible\"). If the query references multiple events (e.g., \"anomalous car visible during a collision\"), focus on the **collision** as the primary event.  \n"
        #     "- **Precision via Temporal Anchoring**: Define the event’s start as the **instant the action initiates** (e.g., vehicle entry into a road) and the end as the **moment the action concludes** (e.g., impact or stoppage). Avoid vague terms like \"during\" or \"after\" without explicit temporal markers.  \n"
        #     "- **Domain-Specific Assumptions**: Assume events are **short-lived** (seconds-scale), **action-driven** (e.g., collisions, anomalies), and **bounded** (e.g., [4.5, 5.3] for a brief anomaly). Avoid overgeneralization by excluding non-essential activity.  \n"
        #     "- **Single-Span Enforcement**: Reinforce the necessity of **one precise interval**, avoiding fragmented or overlapping ranges. If the query implies multiple events, select the **most temporally defined** one (e.g., collision over anomalous visibility).  \n"
        #     "\n"
        #     "**Critical Corrections Based on Feedback:**  \n"
        #     "- **Avoid Broad Spans**: Do not include pre-event (e.g., [0.0, 12.0]) or post-event activity. For example, if the ground truth is [1.6, 5.7], the output must align with this range, not extend beyond it.  \n"
        #     "- **Prioritize Core Event Timeline**: For multi-part events (e.g., lateral movement + collision), focus on the **primary action** (e.g., collision) as the event’s timeline. Exclude secondary actions unless explicitly stated as the event.  \n"
        #     "- **Exact Temporal Alignment**: Ensure the span matches the **ground truth’s exact start and end**. For instance, if the collision begins at 4.4 and ends at 7.2, the output must be [4.4, 7.2], not [5.2, 6.8] or [4.4, 8.2].  \n"
        #     "\n"
        #     "**Niche Domain-Specific Guidance:**  \n"
        #     "- **Event Boundaries**: Assume the event’s start is the **instant of action initiation** (e.g., vehicle entry into a road, brake application) and the end is the **moment the action concludes** (e.g., impact, complete stop). Use domain knowledge to infer these boundaries if explicit markers are absent.  \n"
        #     "- **Primary Event Over Ancillary Details**: If the query mentions multiple events (e.g., \"anomalous car visible during a collision\"), focus on the **collision** as the primary event. Ancillary details like \"anomalous car visible\" are **not** the target unless explicitly stated as the event.  \n"
        #     "- **Short-Lived Events**: Assume all events are **bounded within seconds** (e.g., [2.6, 7.0] for a collision). Avoid spans exceeding 5 seconds unless explicitly justified by the query or ground truth.  \n"
        #     "- **Zero-Shot Temporal Grounding**: If the query lacks explicit temporal markers, rely on **domain-specific assumptions** (e.g., collisions occur within 3–8 seconds) and **ground truth alignment** to infer precise start/end times."
        # )

    # # if dataset_name == "egosurgery":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The video is a segmented surgical procedure with distinct phases (e.g., incision, closure, dissection, suction, hemostasis). Each phase has a predefined [start, end] timestamp range.  \n"
        #     "**Task**: Identify the [start, end] timestamp span that aligns with the query's reference to a specific instrument/action, procedural phase, or action within a phase.  \n"
        #     "**Key Instructions**:  \n"
        #     "1. **Map to phase or instrument**:  \n"
        #     "   - *If the query explicitly mentions a phase* (e.g., \"during incision\"), **use the known [start, end] interval** for that phase.  \n"
        #     "   - *If the query mentions an instrument/action* (e.g., \"Tweezers,\" \"Mouth Gag\"), **check the domain knowledge** to map it to its canonical phase (e.g., Tweezers → closure, Suction Cannula → suction). **If no mapping exists**, **assume the phase explicitly mentioned in the query** (e.g., \"Scissors during closure\" → closure phase).  \n"
        #     "   - *If the query refers to an action within a phase* (e.g., \"Mouth Gag during incision\"), **map it to the phase’s timestamp range**.  \n"
        #     "2. **Prioritize precision**: Always return the **exact phase-specific interval** (e.g., [22.6, 38.6] for incision) **or the instrument’s mapped phase interval** (e.g., Tweezers → [5.6, 19.6]). Avoid guessing or using broad ranges like [0.0, X].  \n"
        #     "3. **Use domain knowledge strictly**:  \n"
        #     "   - *Instruments/Actions → Phases*:  \n"
        #     "     - Tweezers → Closure  \n"
        #     "     - Suction Cannula → Suction  \n"
        #     "     - Mouth Gag → Incision  \n"
        #     "     - Dissection → Dissection  \n"
        #     "     - Hemostasis → Hemostasis  \n"
        #     "   - *Phases → Timestamps*:  \n"
        #     "     - Incision: [22.6, 38.6]  \n"
        #     "     - Closure: [5.6, 19.6]  \n"
        #     "     - Dissection: [42.72, 54.72]  \n"
        #     "     - Suction: [12.3, 28.3]  \n"
        #     "     - Hemostasis: [30.4, 40.4]  \n"
        #     "4. **Handle nested actions**: If the query refers to an action within a phase (e.g., \"Other hands left during hemostasis\"), map it to the phase’s timestamp range.  \n"
        #     "5. **Output exactly one span**: Return the [start, end] of the phase or action’s **predefined interval**, not a guessed range.  \n"
        #     "**Example**:  \n"
        #     "- Query: \"Scalpel during incision\" → Output: [22.6, 38.6] (incision phase, as \"Scalpel\" has no explicit mapping).  \n"
        #     "- Query: \"Other hands left during hemostasis\" → Output: [30.4, 40.4] (hemostasis phase).  \n"
        #     "- Query: \"Needle Holders during closure\" → Output: [5.6, 19.6] (closure phase, as \"Needle Holders\" has no explicit mapping).  \n"
        #     "**Critical Clarification**: If the query contains no explicit phase or instrument, **do not infer**; instead, **return the entire video’s timestamp range** [0.0, X] (where X is the final timestamp of the last phase)."
        # )

    # # if dataset_name == "enigma":
    #     return (
    #         f"Give you a textual query: {query}\n"
    #         "When does the described content occur in the video?\n"
    #         "Please return the timestamp in seconds. Output format:[start, end]"
    #     )

    # # if dataset_name == "meccano":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "Analyze the video's timeline to identify the exact time interval [start, end] in seconds where the described action/event occurs.  \n"
        #     "**Key instructions:**  \n"
        #     "1. Output **only one** time span in the format [start, end], with start ≤ end.  \n"
        #     "2. Ensure timestamps are in **seconds** (e.g., [25.839, 26.589]).  \n"
        #     "3. Do not assume prior knowledge of the video; derive the span solely from the video's content.  \n"
        #     "4. If the action spans multiple segments, return the **union interval** (e.g., [25.839, 27.106] for overlapping events).  \n"
        #     "5. Avoid defaulting to [0.0, X] or [X, Y] without explicit grounding in the video."
        # )

    # # if dataset_name == "mouse":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The query describes a specific behavior of a mouse (e.g., \"repeatedly scratching itself with its hind paw\" or \"grooming its body with its forepaws\"). Your task is to identify the exact time interval [start, end] in seconds when this behavior occurs in the video.  \n"
        #     "\n"
        #     "### Key Instructions:  \n"
        #     "1. **Zero-shot Temporal Grounding**: The behavior may not have been explicitly trained for, so rely on contextual clues (e.g., motion patterns, body part involvement) to infer the correct time span.  \n"
        #     "2. **Single Span Output**: Return **exactly one** [start, end] interval in seconds, matching the video's timestamp format (e.g., [161.82, 167.45]).  \n"
        #     "3. **Precision Matters**: Ensure the span aligns with the behavior's duration. For example, if the query mentions \"repeated grooming,\" the span should cover all instances of the behavior in sequence.  \n"
        #     "4. **Domain-Specific Focus**: Prioritize behaviors involving specific body parts (forepaws, hind paw) and actions (scratching, grooming) as described. Avoid extrapolating beyond the query.  \n"
        #     "5. **Format Strictly**: Output only the [start, end] interval, with no additional text or explanations."
        # )

    # # if dataset_name == "multisports":
    #     return (
    #         f"Give you a textual query: {query}\n"
    #         "When does the described content occur in the video?\n"
    #         "Please return the timestamp in seconds. "
    #         "Output format:"
    #         "[start, end]"
    #     )

    # # if dataset_name == "uca":
        # return (
        #     f"Give you a textual query: {query}  \n"
        #     "The task is to determine the exact time interval [start, end] in seconds when the described content occurs in the video.  \n"
        #     "**Instructions for the model**:  \n"
        #     "1. **Anchor on the first explicit action**: Identify the **first action or state** in the query (e.g., \"wiped the table,\" \"lowered his head\") as the **primary temporal anchor**. Use this as the starting point for the interval.  \n"
        #     "2. **Sequence alignment**: Extend the interval to include **all subsequent actions or states** in the query (e.g., \"stepped forward,\" \"slapped the old man\"). Ensure the interval spans the **entire sequence** of actions, even if they are described in a single sentence.  \n"
        #     "3. **Prioritize unambiguous visual cues**: Focus on **specific, distinct descriptors** (e.g., \"little boy in blue,\" \"clerk in a convenience store\") to narrow the time window. If multiple cues are present, select the one most likely to correspond to a **clear temporal marker** (e.g., a unique object placement or action sequence).  \n"
        #     "4. **Enforce minimal temporal span**: Output the **shortest possible interval** that fully includes **all elements of the query**. Avoid overextending the interval to unrelated events. If the query describes a brief interaction (e.g., \"talked to someone\"), prioritize a **narrow span** reflecting the implied brevity.  \n"
        #     "5. **Avoid extrapolation**: Do **not assume** the event occurs at the beginning or end of the video unless explicitly stated. Use the **phrasing** of the query (e.g., \"started to wipe\" vs. \"was wiping\") to infer the likely timing.  \n"
        #     "6. **Format strictly**: Output **only the interval** in [start, end] format, using **seconds with one decimal place** (e.g., [42.5, 57.0]). Do not include any additional text or explanations."
        # )


    if model_family == "eagle":
        return (
            f"Give you a textual query: {query}\n"
            "When does the described content occur in the video?\n"
            "Please return the timestamp in seconds. "
            "Output format:"
            "[start, end]"
        )

    if model_family == "llava_st":
        # official example: "Give you a textual query: 'person takes a laptop from the shelf'. When does the described content occur in the video? Please return the start and end timestamps."
        return (
            f"Give you a textual query: '{query}'. "
            "When does the described content occur in the video? "
            "Please return the start and end timestamps."
        )


    raise ValueError("Invalid model inputs")

