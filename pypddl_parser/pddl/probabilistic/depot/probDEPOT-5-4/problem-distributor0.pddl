(define (problem depot-5-4-distributor0) (:domain depot-5-4-distributor0)
(:objects
	truck0 - truck
	depot0 - depot
	distributor0 - distributor
	distributor1 - distributor
	crate0 - crate
	crate1 - crate
	crate2 - crate
	pallet2 - pallet
	pallet1 - pallet
	pallet0 - pallet

	(:private
		hoist1 - hoist
	)
)
(:init
	(at hoist1 distributor0)
	(available distributor0 hoist1)

	(at truck0 depot0)
	(at pallet0 depot0)
	(at pallet1 distributor0)
	(at pallet2 distributor1)
	(at crate0 depot0)
	(at crate1 distributor0)
	(at crate2 distributor1)
	(on crate0 pallet0)
	(on crate1 pallet1)
	(on crate2 pallet2)
	(clear crate0)
	(clear crate1)
	(clear crate2)
)
(:goal
	(and
		(on crate0 pallet1)
		(on crate1 pallet2)
		(on crate2 pallet0)
	)
)
)