from typing import Any, Optional, Union, cast

import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import PAD, PUNCTUATION_SYMBOLS, SYMBOLS


def get_net_g(
    model_path: str, version: str, device: str, hps: HyperParameters
) -> Union[SynthesizerTrn, SynthesizerTrnJPExtra]:
    if version.endswith("JP-Extra"):
        logger.info("Using JP-Extra model")
        net_g = SynthesizerTrnJPExtra(
            n_vocab=len(SYMBOLS),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        ).to(device)
    else:
        logger.info("Using normal model")
        net_g = SynthesizerTrn(
            n_vocab=len(SYMBOLS),
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        ).to(device)
    net_g.state_dict()
    _ = net_g.eval()
    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors"):
        _ = utils.safetensors.load_safetensors(model_path, net_g, True, device=device)
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    use_jp_extra = hps.version.endswith("JP-Extra")
    norm_text, phone, tone, word2ph = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        use_jp_extra=use_jp_extra,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
    )
    del word2ph
    assert bert_ori.shape[-1] == len(phone), phone

    if language_str == Languages.ZH:
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.JP:
        bert = torch.zeros(1024, len(phone))
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone))
    elif language_str == Languages.EN:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(1024, len(phone))
        en_bert = bert_ori
    else:
        raise ValueError("language_str should be ZH, JP or EN")

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, en_bert, phone, tone, language


def extract_phone_durations(
    attn: torch.Tensor,
    phone_ids: "list[int] | torch.Tensor",
    hps: HyperParameters,
    blank_mode: str = "boundary_and_punctuation",
    pause_label: str = "pau",
    punctuation_as_pause: bool = True,
    unvoiced_vowel_flags: "Optional[list[bool]]" = None,
) -> "list[tuple[str, int, int]]":
    """
    net_g.infer() が返す attn（音素→フレームのアラインメントパス）から、
    各音素の開始・終了時刻（100ns単位、HTS/Julius式 .lab フォーマット準拠）を算出する。

    attn の形状は [1, 1, t_y, t_x] （t_y: 出力フレーム数, t_x: 音素数）で、
    各列（x軸）にちょうど1個の1が並ぶ対角的な0/1行列になっている。
    列方向（dim=2, フレーム軸）に総和を取ると、その音素が占有したフレーム数が得られる。

    重要な注意点（PAD の二重構造）: Style-Bert-VITS2 では、
    - g2p（clean_text_with_given_phone_tone / g2p.py）が phone 列の最初と最後に
      すでに "_"（PAD と同じ文字・同じ symbol id）を1つずつ追加している
    - さらに add_blank=True の場合、get_text() が commons.intersperse() で
      すべてのトークン（この "_" も含む）の前後に PAD を挿入する
    という2段階の処理があるため、文頭・文末には PAD が連続して複数個並ぶ
    （例: [PAD, PAD, PAD, k, PAD, o, ..., PAD, PAD, PAD]）。単純に「最初の
    PAD だけ」「最後の PAD だけ」を境界として扱うと、連続する PAD 群の
    残りの部分が中途半端に merge されて時刻計算がズレる/一部の区間が
    欠落する不具合が起きる。そのため本関数はまず連続する pause 扱いトークン
    （PAD、および punctuation_as_pause=True の場合は句読点も含む）を1つの
    「pause ラン（run）」としてグルーピングし、そのラン単位で扱いを判定する。

    また、句読点（PUNCTUATION_SYMBOLS: "!", "?", "…", ",", ".", "'", "-",
    "SP", "UNK"）は PAD とは別の実 symbol id を持つため、そのままでは
    "," や "." といった文字がラベルとして出力される。punctuation_as_pause=True
    （デフォルト）ではこれらも PAD と同様に pause 扱いにまとめ、pause_label
    として出力する。

    Args:
        attn: SynthesizerTrn(JPExtra).infer() の戻り値タプルの2番目の要素（output[1]）。
        phone_ids: get_text() が返す phone 列（add_blank 適用後、PAD が音素間に挿入された状態）。
                   list[int] でも 1D/2D の torch.Tensor でも可。
        hps: hop_length と sampling_rate を読むために使用。
        blank_mode: pause 扱いとなる区間（PAD ラン、punctuation_as_pause=True の
            場合は句読点も含む）の扱い方。
            - "boundary_and_punctuation"（デフォルト）: 文頭・文末の pause ラン、
              および句読点を含む pause ランを pause_label 行として残す。
              音素間の技術的な PAD（句読点を含まないもの）のみ左右の実音素へ
              吸収させて消す。句読点による息継ぎ・間は明示的に残しつつ、
              音素間には無駄な pau 行が入らない、実用上もっとも自然な形。
            - "boundary_only": 文頭・文末の pause ランのみ残す。句読点も含め、
              文中の pause ランはすべて左右の実音素へ吸収させて消す。
            - "all": すべての pause ランを、ラン単位で pause_label という
              独立した行として残す。句読点を含まない音素間の PAD にも
              短い pau 行が入る（デバッグ・検証用途向け）。
            - "merge": すべての pause 区間を左右の実音素へ吸収させ、pause 行を
              一切出力しない（文頭・文末の無音も実音素側へ吸収されるため、
              最初の音素が実際の発声より早く始まる/最後の音素が遅く終わる
              形になる）。
        pause_label: pause 区間に付けるラベル（blank_mode="merge" のときは未使用）。
        punctuation_as_pause: True（デフォルト）の場合、句読点トークン
            （PUNCTUATION_SYMBOLS）も PAD と同様に pause 扱いにする。False の
            場合、句読点は実音素と同じように、そのままの文字（"," "." など）
            がラベルとして出力される（この場合 blank_mode="boundary_and_punctuation"
            は "boundary_only" と同じ結果になる）。
        unvoiced_vowel_flags: 実音素（pause/句読点を除く）ごとに、その母音が
            無声化されていたかどうかを表す bool のリスト。
            style_bert_vits2.nlp.japanese.g2p.get_unvoiced_vowel_flags() で
            テキストから独立に算出したものを渡す想定。None の場合は無声化判定を
            行わず、母音ラベルは常に小文字のまま出力する（従来どおりの挙動）。
            長さは実音素の個数と一致している必要がある（一致しない場合は
            無声化判定をスキップし、警告をログに出す）。
            True の位置に対応する実音素ラベルが母音（a/i/u/e/o）であれば、
            出力時のみそのラベルを大文字（A/I/U/E/O）に変える
            （SYMBOLS/モデルの音素IDには一切影響しない、表示上の変換）。

    Returns:
        (phoneme_label, start_100ns, end_100ns) のタプルのリスト。
    """

    if attn.dim() == 4:
        # [b, 1, t_y, t_x] -> [t_y, t_x]  （b=1 前提。バッチ推論はサポート外）
        attn = attn[0, 0]
    elif attn.dim() == 3:
        # [1, t_y, t_x] -> [t_y, t_x]
        attn = attn[0]

    valid_blank_modes = ("boundary_and_punctuation", "boundary_only", "all", "merge")
    if blank_mode not in valid_blank_modes:
        raise ValueError(
            f"blank_mode must be one of {valid_blank_modes}, got {blank_mode!r}"
        )

    # 各音素（列）が占めるフレーム数 = その列の総和
    frame_counts = attn.sum(dim=0).long().cpu().tolist()  # len == t_x

    if isinstance(phone_ids, torch.Tensor):
        phone_ids = phone_ids.view(-1).cpu().tolist()

    n = len(phone_ids)
    if len(frame_counts) != n:
        raise ValueError(
            f"attn の音素次元数({len(frame_counts)})と phone_ids の長さ"
            f"({n})が一致しません。skip_start/skip_end 適用後の"
            f"phone_ids を渡しているか確認してください。"
        )

    pad_id = SYMBOLS.index(PAD)
    punctuation_ids = (
        {SYMBOLS.index(p) for p in PUNCTUATION_SYMBOLS if p in SYMBOLS}
        if punctuation_as_pause
        else set()
    )
    pause_like_ids = {pad_id} | punctuation_ids

    # 連続する pause 扱いトークン（PAD および、有効なら句読点）を「ラン」として
    # グルーピングする。各ランは (start_idx, end_idx_exclusive, is_pause,
    # has_punctuation)。has_punctuation はそのラン内に句読点トークンが
    # 1つでも含まれるかどうか（"boundary_and_punctuation" モードの判定に使う）。
    runs: list[tuple[int, int, bool, bool]] = []
    i = 0
    while i < n:
        is_pause = phone_ids[i] in pause_like_ids
        j = i + 1
        while j < n and (phone_ids[j] in pause_like_ids) == is_pause and is_pause:
            j += 1
        has_punct = is_pause and any(
            phone_ids[k] in punctuation_ids for k in range(i, j)
        )
        runs.append((i, j, is_pause, has_punct))
        i = j

    pause_run_indices = [
        ri for ri, (_, _, is_pause, _) in enumerate(runs) if is_pause
    ]

    if blank_mode == "all":
        keep_pause_runs = set(pause_run_indices)
    elif blank_mode == "boundary_only":
        keep_pause_runs = set()
        if pause_run_indices:
            keep_pause_runs.add(pause_run_indices[0])
            keep_pause_runs.add(pause_run_indices[-1])
    elif blank_mode == "boundary_and_punctuation":
        keep_pause_runs = set()
        if pause_run_indices:
            keep_pause_runs.add(pause_run_indices[0])
            keep_pause_runs.add(pause_run_indices[-1])
        for ri, (_, _, is_pause, has_punct) in enumerate(runs):
            if is_pause and has_punct:
                keep_pause_runs.add(ri)
    else:  # "merge"
        keep_pause_runs = set()

    frame_to_100ns = (hps.data.hop_length / hps.data.sampling_rate) * 1e7  # 100ns単位

    # unvoiced_vowel_flags は実音素（pause/句読点を除く）の出現順に対応する。
    # 実際に使うかどうかは、実音素の個数と一致するかどうかで判定する
    # （テキストの取り違え等でズレている場合に誤ったラベルを大文字化しないため）。
    num_real_phones = sum(
        1 for (_, _, is_pause, _) in runs if not is_pause
    )
    use_unvoiced_flags = unvoiced_vowel_flags is not None and len(
        unvoiced_vowel_flags
    ) == num_real_phones
    if unvoiced_vowel_flags is not None and not use_unvoiced_flags:
        logger.warning(
            f"unvoiced_vowel_flags の長さ({len(unvoiced_vowel_flags)})が実音素数"
            f"({num_real_phones})と一致しないため、無声化母音の大文字化をスキップします。"
        )
    real_phone_idx = 0

    result: list[tuple[str, int, int]] = []
    cursor_frames = 0.0
    for ri, (start, end, is_pause, _has_punct) in enumerate(runs):
        run_frames = sum(frame_counts[start:end])

        if is_pause and ri not in keep_pause_runs:
            # 実音素に吸収させる pause ラン。直前に確定済みの行があれば
            # その終了時刻を後ろへ延ばす形で譲渡する（直前が無ければ
            # cursor_frames をそのまま進めて次の行の開始を遅らせる）。
            if result:
                label, s, _old_e = result[-1]
                new_cursor = cursor_frames + run_frames
                new_e = round(new_cursor * frame_to_100ns)
                result[-1] = (label, s, new_e)
            cursor_frames += run_frames
            continue

        if run_frames == 0:
            continue

        start_100ns = round(cursor_frames * frame_to_100ns)
        cursor_frames += run_frames
        end_100ns = round(cursor_frames * frame_to_100ns)

        if is_pause:
            result.append((pause_label, start_100ns, end_100ns))
        else:
            # このランは通常、実音素1個分（intersperse の隙間には pause 扱いの
            # トークンしか入らないため、is_pause=False のランは基本的に長さ1）。
            # 万一長さ2以上の非pause ランが来ても、最初のIDのラベルで代表させる。
            pid = phone_ids[start]
            label = SYMBOLS[pid] if isinstance(pid, int) else str(pid)

            if use_unvoiced_flags and label in ("a", "i", "u", "e", "o"):
                assert unvoiced_vowel_flags is not None
                if unvoiced_vowel_flags[real_phone_idx]:
                    label = label.upper()

            result.append((label, start_100ns, end_100ns))
            real_phone_idx += 1

    return result


def write_lab_file(
    entries: "list[tuple[str, int, int]]",
    output_path: str,
) -> None:
    """
    extract_phone_durations() の出力を HTS/Julius 互換の .lab ファイルとして書き出す。

    フォーマット: 1行 = "<開始(100ns)> <終了(100ns)> <音素ラベル>"
    """

    with open(output_path, "w", encoding="utf-8") as f:
        for label, start_100ns, end_100ns in entries:
            f.write(f"{start_100ns} {end_100ns} {label}\n")


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,  # In the original Bert-VITS2, its speaker_name: str, but here it's id
    language: Languages,
    hps: HyperParameters,
    net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra],
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    return_phone_durations: bool = False,
    phone_durations_blank_mode: str = "boundary_only",
    pause_label: str = "pau",
    punctuation_as_pause: bool = True,
) -> "NDArray[Any] | tuple[NDArray[Any], list[tuple[str, int, int]]]":
    is_jp_extra = hps.version.endswith("JP-Extra")
    bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        bert = bert[:, 3:]
        ja_bert = ja_bert[:, 3:]
        en_bert = en_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        bert = bert[:, :-2]
        ja_bert = ja_bert[:, :-2]
        en_bert = en_bert[:, :-2]

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        en_bert = en_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        if is_jp_extra:
            output = cast(SynthesizerTrnJPExtra, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                ja_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )
        else:
            output = cast(SynthesizerTrn, net_g).infer(
                x_tst,
                x_tst_lengths,
                sid_tensor,
                tones,
                lang_ids,
                bert,
                ja_bert,
                en_bert,
                style_vec=style_vec_tensor,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
            )

        audio = output[0][0, 0].data.cpu().float().numpy()

        phone_durations: Optional[list[tuple[str, int, int]]] = None
        if return_phone_durations:
            attn = output[1]

            # 無声化母音（本来 pyopenjtalk が A/E/I/O/U として返すもの）を
            # .lab 出力でのみ大文字化するためのフラグ列。given_phone が
            # 指定されている場合は、読みが text と食い違い得るため計算しない
            # （extract_phone_durations 側の長さチェックにより安全側に倒れる）。
            unvoiced_vowel_flags: Optional[list[bool]] = None
            if given_phone is None and language == Languages.JP:
                try:
                    from style_bert_vits2.nlp.japanese.g2p import (
                        get_unvoiced_vowel_flags,
                    )
                    from style_bert_vits2.nlp.japanese.normalizer import (
                        normalize_text,
                    )

                    # g2p() 側も normalize_text() 済みのテキストに対して実行
                    # されているため、無声化フラグの計算にも同じ正規化後の
                    # テキストを使う（そうしないと文字位置がズレて誤対応する）。
                    unvoiced_vowel_flags = get_unvoiced_vowel_flags(
                        normalize_text(text)
                    )
                except Exception:
                    logger.warning(
                        "無声化母音フラグの計算に失敗したため、"
                        ".lab 出力での無声化母音の大文字化をスキップします。",
                        exc_info=True,
                    )
                    unvoiced_vowel_flags = None

            # x_tst は skip_start/skip_end 適用後、かつ CPU 上の元テンソルなので
            # そのまま phone_ids として使う（extract_phone_durations 側で
            # to-list 変換される）。
            phone_durations = extract_phone_durations(
                attn,
                x_tst[0],
                hps,
                blank_mode=phone_durations_blank_mode,
                pause_label=pause_label,
                punctuation_as_pause=punctuation_as_pause,
                unvoiced_vowel_flags=unvoiced_vowel_flags,
            )

        del (
            x_tst,
            tones,
            lang_ids,
            bert,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            en_bert,
            style_vec,
        )  # , emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_phone_durations:
            assert phone_durations is not None
            return audio, phone_durations
        return audio
