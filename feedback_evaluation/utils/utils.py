class AverageMeter(object):
    def __init__(self):
        self.hinge_loss = 0
        self.pure_loss = 0
        self.l1_term = 0
        self.pos_prior = 0
        self.neg_prior = 0

        self.hinge_loss_sum = 0
        self.pure_loss_sum = 0
        self.l1_term_sum = 0
        self.pos_prior_sum = 0
        self.neg_prior_sum = 0

        self.hinge_loss_avg = 0
        self.pure_loss_avg = 0
        self.l1_term_avg = 0
        self.pos_prior_avg = 0
        self.neg_prior_avg = 0
        
        self.count = 0

    def reset(self):
        self.hinge_loss = 0
        self.pure_loss = 0
        self.l1_term = 0
        self.pos_prior = 0
        self.neg_prior = 0

        self.hinge_loss_sum = 0
        self.pure_loss_sum = 0
        self.l1_term_sum = 0
        self.pos_prior_sum = 0
        self.neg_prior_sum = 0

        self.hinge_loss_avg = 0
        self.pure_loss_avg = 0
        self.l1_term_avg = 0
        self.pos_prior_avg = 0
        self.neg_prior_avg = 0
        
        self.count = 0

    def update(self, hinge_loss, pure_loss, l1_term, pos_prior, neg_prior):
        self.hinge_loss = hinge_loss
        self.pure_loss = pure_loss
        self.l1_term = l1_term
        self.pos_prior = pos_prior
        self.neg_prior = neg_prior

        self.hinge_loss_sum += hinge_loss
        self.pure_loss_sum += pure_loss
        self.l1_term_sum += l1_term
        self.pos_prior_sum += pos_prior
        self.neg_prior_sum += neg_prior

        self.count += 1

        self.hinge_loss_avg = self.hinge_loss_sum / self.count
        self.pure_loss_avg = self.pure_loss_sum / self.count
        self.l1_term_avg = self.l1_term_sum / self.count
        self.pos_prior_avg = self.pos_prior_sum / self.count
        self.neg_prior_avg = self.neg_prior_sum / self.count