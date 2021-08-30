(define (problem depot-4-3-depot0) (:domain depot-4-3-depot0)
(:objects
	truck0 - truck
	depot0 - depot
	distributor0 - distributor
	crate0 - crate
	crate1 - crate
	pallet1 - pallet
	pallet0 - pallet

	(:private
		hoist0 - hoist
	)
)
(:init
    (at hoist0 depot0)
	(available depot0 hoist0)

	(at truck0 depot0)
	(at pallet0 depot0)
	(at pallet1 distributor0)
	(at crate0 depot0)
	(at crate1 distributor0)
	(on crate0 pallet0)
	(on crate1 pallet1)
	(clear crate0)
	(clear crate1)
)
(:goal
	(and
		(on crate0 pallet1)
		(on crate1 pallet0)
	)
)
)