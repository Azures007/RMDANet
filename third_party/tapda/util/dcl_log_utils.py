"""Console log formatting helpers for DCL training loops."""


def format_train_status(ep, batch_idx, data1, train_loader, loss1, loss2, loss_dis,
                        bnm_loss_a, contrastive_loss_s, contrastive_loss_t,
                        target_contrastive_stats, args):
    """Format console training status without changing any training values."""
    parts = [
        'Train Ep: {} [{}/{} ({:.0f}%)]'.format(
            ep,
            batch_idx * len(data1),
            args.batch_size * len(train_loader),
            100. * batch_idx / len(train_loader)
        ),
        'Loss1: {:.6f}'.format(loss1.item()),
        'Loss2: {:.6f}'.format(loss2.item()),
        'Dis: {:.6f}'.format(loss_dis.item()),
        'BNM_A: {:.6f}'.format(bnm_loss_a.item()),
        'ConS: {:.6f}'.format(contrastive_loss_s.item()),
        'ConT: {:.6f}'.format(contrastive_loss_t.item()),
    ]

    if args.log_target_contrast_detail:
        parts.append(
            'SelT: {} ClsT: {} ThrT: {:.2f} ModeT: {}'.format(
                target_contrastive_stats['selected'],
                target_contrastive_stats['selected_classes'],
                target_contrastive_stats['threshold'],
                target_contrastive_stats['selection_mode']
            )
        )
    if args.log_target_quota_detail:
        parts.append(
            'QuotaT: {} QAvgT: {:.2f} QMaxT: {} QDiffT: {:.2f} QScarT: {:.2f}'.format(
                target_contrastive_stats['quota_mode'],
                target_contrastive_stats['quota_avg'],
                target_contrastive_stats['quota_max'],
                target_contrastive_stats['quota_difficulty_mean'],
                target_contrastive_stats['quota_scarcity_mean']
            )
        )
    if args.log_target_prototype_detail:
        parts.append(
            'ProtoT: {} PCandT: {}/{} PKeepT: {:.2f} PClsT: {} PScoreT: {} PMrgT: {:.3f} PSMeanT: {:.3f}'.format(
                target_contrastive_stats['prototype_filter'],
                target_contrastive_stats['prototype_after'],
                target_contrastive_stats['prototype_before'],
                target_contrastive_stats['prototype_keep_rate'],
                target_contrastive_stats['prototype_valid_classes'],
                target_contrastive_stats['prototype_scoring'],
                target_contrastive_stats['prototype_margin_mean'],
                target_contrastive_stats['prototype_score_mean']
            )
        )

    return '\t'.join(parts)
